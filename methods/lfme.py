import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models
import random
# from domainbed.lib import wide_resnet
import copy


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride,
                    bias=True), )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    """Wide Resnet with the softmax layer chopped off"""
    def __init__(self, input_shape, depth, widen_factor, dropout_rate):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(input_shape[0], nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

        self.n_outputs = nStages[3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        return out[:, :, 0, 0]

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0)
        m.bias.data.fill_(1)

    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

    if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0)

class MappingNetwork(torch.nn.Module):
    def __init__(self, depth=5):
        super().__init__()
        self.depth = depth
        self.weight1 = nn.ParameterList()
        self.bias1 = nn.ParameterList()
        self.weight2 = nn.ParameterList()
        self.bias2 = nn.ParameterList()
        self.weight3 = nn.ParameterList()
        self.bias3 = nn.ParameterList()
        self.weight4 = nn.ParameterList()
        self.bias4 = nn.ParameterList()
        for i in range(depth):
            self.weight1.append(nn.Parameter(torch.ones((64,56,56))))
            self.bias1.append(nn.Parameter(torch.zeros((64,56,56))))

            self.weight2.append(nn.Parameter(torch.ones((128,28,28))))
            self.bias2.append(nn.Parameter(torch.zeros((128,28,28))))

            self.weight3.append(nn.Parameter(torch.ones((256,14,14))))
            self.bias3.append(nn.Parameter(torch.zeros((256,14,14))))

            self.weight4.append(nn.Parameter(torch.ones((512, 7, 7))))
            self.bias4.append(nn.Parameter(torch.zeros((512, 7, 7))))

        self.relu = nn.ReLU(inplace=True)

    def fea1(self, x):
        for i in range(self.depth-1):
            x = self.relu(self.weight1[i] * x + self.bias1[i])
        x = self.weight1[i+1] * x + self.bias1[i+1]
        return x

    def fea2(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight2[i] * x + self.bias2[i])
        x = self.weight2[i + 1] * x + self.bias2[i + 1]
        return x

    def fea3(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight3[i] * x + self.bias3[i])
        x = self.weight3[i + 1] * x + self.bias3[i + 1]
        return x

    def fea4(self, x):
        for i in range(self.depth-1):
            x = self.relu(self.weight4[i] * x + self.bias4[i])
        x = self.weight4[i+1] * x + self.bias4[i+1]
        return x


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Adaparams(nn.Module):
    def __init__(self, depth=10):
        super(Adaparams, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.depth = depth
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()
        for i in range(depth):
            self.weight.append(nn.Parameter(torch.ones(512)))
            self.bias.append(nn.Parameter(torch.zeros(512)))

    def forward(self, x):
        for i in range(self.depth-1):
            x = self.relu(self.weight[i] * x + self.bias[i])
        x = self.weight[i+1] * x + self.bias[i+1]
        return x
        

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        #self.avgpool = nn.AvgPool2d(7,stride=1)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams.get('resnet18', False):
            self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class ResNet_ITTA(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet_ITTA, self).__init__()
        if hparams.get('resnet18', False):
            self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.n_outputs = 2048

        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        self.network.fc = Identity()
        self.isaug = True
        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.eps = 1e-6

    def mixstyle(self, x):
        alpha = 0.1
        beta = torch.distributions.Beta(alpha, alpha)
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        lmda = beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix

    def fea_forward(self, x):
        x = self.fea3(x)
        x = self.fea4(x)

        x = self.flat(x)
        return x

    def fea2(self, x, aug_x):
        x = self.network.layer2(x)
        aug_x = self.network.layer2(aug_x)
        if not self.isaug:
            aug_x = self.mixstyle(aug_x)
        return x, aug_x

    def fea3(self, x):
        x = self.network.layer3(x)
        return x

    def fea4(self, x):
        x = self.network.layer4(x)
        return x

    def flat(self, x):
        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.network.fc(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        if random.random() > 0.5:
            self.isaug = True
            aug_x = self.mixstyle(x)
        else:
            self.isaug = False
            aug_x = x

        return x, aug_x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError
    
# def Featurizer_OTHMix(input_shape, hparams, part='trunk'):
#     if input_shape[1:3] == (224, 224):
#         if part == 'base':
#             return ResNet_base(input_shape, hparams)
#         elif part == 'trunk':
#             return ResNet_trunk(input_shape, hparams)
#         else:
#             raise NotImplementedError
#     else:
#         if part == 'base':
#             return MNIST_base(input_shape)
#         elif part == 'trunk':
#             return MNIST_trunk(input_shape)
#         else:
#             raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

###########################################
class ResNet_base(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet_base, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        del self.network.layer2
        del self.network.layer3
        del self.network.layer4
        del self.network.avgpool

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        # x = self.network.layer2(x)
        # x = self.network.layer3(x)
        # x = self.network.layer4(x)
        #
        # x = self.network.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.network.fc(x)

        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNet_trunk(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet_trunk, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        del self.network.conv1
        del self.network.bn1
        del self.network.relu
        del self.network.maxpool
        del self.network.layer1
        #del self.network.layer2
        #del self.network.layer3

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        # x = self.network.conv1(x)
        # x = self.network.bn1(x)
        # x = self.network.relu(x)
        # x = self.network.maxpool(x)
        #
        # x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)

        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.network.fc(x)
        x = self.dropout(x)

        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
    

class LFME(Algorithm):
    """
    Learning from Multiple Experts for Domain Generalization
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LFME, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

        def make_network():
            featurizer = Featurizer(input_shape, hparams)
            classifier = Classifier(featurizer.n_outputs, num_classes, hparams.get("nonlinear_classifier", False),)
            return nn.Sequential(featurizer, classifier)

        # 반드시 ModuleList를 사용
        self.experts = nn.ModuleList([make_network() for _ in range(num_domains)])

        # 실제 inference에 사용하는 모델
        self.target = make_network()

    def forward_expert(self, domain_idx, x):
        return self.experts[domain_idx](x)

    def forward(self, x):
        """Target model logits."""
        return self.target(x)

    def predict(self, x):
        return self.forward(x)
			
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        expert = torch.zeros(all_y.shape[0], self.num_classes).to('cuda')
        for i in range(self.expert_number-1):
            mmbatch = minibatches[i]
            part_x, part_y = mmbatch[0], mmbatch[1]
            result_expert = self.network[i](part_x)
            loss = F.cross_entropy(result_expert, part_y)
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
            index, end = (i) * part_y.shape[0], (i + 1) * part_y.shape[0]
            expert[index:end, :] = F.softmax(result_expert, dim=1)

        result_target = self.network[-1](all_x)
        loss_cla = F.cross_entropy(result_target, all_y)
        loss_guid = self.MSEloss(result_target, expert.detach())
        loss = loss_cla + loss_guid * self.hparams['lfe_reg']
        self.optimizer[-1].zero_grad()
        loss.backward()
        self.optimizer[-1].step()
        return {'loss': loss.item()}

    # def predict(self, x):
    #    return self.network[-1](x)