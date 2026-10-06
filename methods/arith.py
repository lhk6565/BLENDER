from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.models import (ResNet18_Weights, ResNet50_Weights, resnet18, resnet50)


# Temporary parameter used by the first-order inner loop.
if not hasattr(Parameter, "fast"):
    Parameter.fast = None


def _active(parameter: Optional[Parameter]):
    if parameter is None:
        return None
    return parameter if parameter.fast is None else parameter.fast


class FastLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, _active(self.weight), _active(self.bias))


class FastConv2d(nn.Conv2d):
    def forward(self, x):
        return F.conv2d(x, _active(self.weight), _active(self.bias), self.stride, self.padding, self.dilation, self.groups)


class FastBatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm with fast affine parameters.

    Running statistics stay in the original module buffers, while the
    affine weight and bias can use temporary fast parameters.
    """

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = (1.0 / float(self.num_batches_tracked))

        bn_training = self.training or (self.running_mean is None and self.running_var is None)

        return F.batch_norm(x,
                            self.running_mean if (not self.training or self.track_running_stats) else None,
                            self.running_var if (not self.training or self.track_running_stats) else None,
                            _active(self.weight),
                            _active(self.bias),
                            bn_training,
                            exponential_average_factor,
                            self.eps,)


def _replace_with_fast_layers(module: nn.Module) -> None:
    """
    Replace Conv2d, BatchNorm2d and Linear modules in-place while preserving
    the pretrained state.
    """

    for name, child in list(module.named_children()):
        replacement = None

        if isinstance(child, nn.Conv2d):
            replacement = FastConv2d(in_channels=child.in_channels,
                                     out_channels=child.out_channels,
                                     kernel_size=child.kernel_size,
                                     stride=child.stride,
                                     padding=child.padding,
                                     dilation=child.dilation,
                                     groups=child.groups,
                                     bias=child.bias is not None,
                                     padding_mode=child.padding_mode)

        elif isinstance(child, nn.BatchNorm2d):
            replacement = FastBatchNorm2d(num_features=child.num_features,
                                          eps=child.eps,
                                          momentum=child.momentum,
                                          affine=child.affine,
                                          track_running_stats=child.track_running_stats)

        elif isinstance(child, nn.Linear):
            replacement = FastLinear(in_features=child.in_features, out_features=child.out_features, bias=child.bias is not None)

        if replacement is None:
            _replace_with_fast_layers(child)
        else:
            replacement.load_state_dict(child.state_dict())
            replacement.train(child.training)
            setattr(module, name, replacement)


class ARITH(nn.Module):
    """
    Independent ARITH baseline:
        ImageNet-pretrained ResNet -> linear closed-set classifier.
    """

    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()

        backbone = backbone.lower()

        if backbone == "resnet50":
            weights = (ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_net = resnet50(weights=weights)

        elif backbone == "resnet18":
            # Available for smoke tests. Use ResNet-50 for reported results.
            weights = (ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_net = resnet18(weights=weights)

        else:
            raise ValueError(f"Unsupported ARITH backbone: {backbone}. Use resnet50 for the paper comparison.")

        feature_dim = int(feature_net.fc.in_features)
        feature_net.fc = nn.Identity()
        _replace_with_fast_layers(feature_net)

        self.backbone = feature_net
        self.classifier = FastLinear(feature_dim, int(num_classes))

        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def clear_fast_weights(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.fast = None


def make_arithmetic_weights(num_source_domains: int) -> List[float]:
    """
    Normalized decreasing arithmetic weights.

    3 source domains:
        [3, 2, 1] / 6 = [1/2, 1/3, 1/6]

    5 source domains:
        [5, 4, 3, 2, 1] / 15
        = [1/3, 4/15, 1/5, 2/15, 1/15]
    """

    num_source_domains = int(num_source_domains)

    if num_source_domains <= 0:
        raise ValueError("num_source_domains must be positive.")

    raw = list(range(num_source_domains, 0, -1))
    denominator = float(sum(raw))

    return [value / denominator for value in raw]


def compute_domain_gradient(model, fast_parameters, x, y):
    """
    Compute the cross-entropy gradient for one source domain at the current
    fast parameters.
    """

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    gradients = torch.autograd.grad(loss, fast_parameters, create_graph=False, retain_graph=False, allow_unused=True)
    gradients = [gradient.detach() if gradient is not None else None for gradient in gradients]

    return loss, logits.detach(), gradients


def inner_sgd_step(model, gradients, inner_lr,):
    """
    One momentum-free SGD inner step:
        theta_fast <- theta_fast - inner_lr * gradient

    The original model parameters are not overwritten.
    """

    fast_parameters: List[torch.Tensor] = []
    displacements: List[Optional[torch.Tensor]] = []

    for parameter, gradient in zip(model.parameters(), gradients):
        current = (parameter if parameter.fast is None else parameter.fast)

        if gradient is None:
            fast_parameters.append(current)
            displacements.append(None)
            continue

        updated = (current - float(inner_lr) * gradient).detach().requires_grad_(True)

        displacement = current.detach() - updated.detach()

        parameter.fast = updated
        fast_parameters.append(updated)
        displacements.append(displacement)

    return fast_parameters, displacements


def accumulate_arithmetic_gradients(model, displacements, weight,):
    """
    Add the arithmetic-weighted inner displacement to the outer gradient.
    """

    weight = float(weight)

    for parameter, displacement in zip(model.parameters(), displacements,):
        if displacement is None:
            continue

        contribution = weight * displacement

        if parameter.grad is None:
            parameter.grad = contribution.clone()
        else:
            parameter.grad.add_(contribution)
