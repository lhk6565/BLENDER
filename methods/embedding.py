import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class cnnBlock(nn.Module):
    def __init__(self, in_channels, conv_dims, **kwargs):
        super().__init__()
        self.C, self.H, self.W = in_channels
        in_conv = in_channels[0]
        kernel_size = kwargs.get('kernel_size', 5)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 2)

        modules = []
        for h_dim in conv_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_conv, h_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2,2)))
            in_conv = h_dim

        self.conv = nn.Sequential(*modules)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return x, (self.C, self.H, self.W)
    
    def _calculate_embed_dim(self):
        dummy = torch.zeros(1, self.C, self.H, self.W)
        dummy = self.conv(dummy)
        self.C, self.H, self.W = dummy.shape[1:]
        embed_dim = self.C * self.H * self.W
        return embed_dim


class RevcnnBlock(nn.Module):
    def __init__(self, rev_conv_dims, in_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        kernel_size = kwargs.get('kernel_size', 5)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 2)

        modules = []
        for i in range(len(rev_conv_dims)-1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(rev_conv_dims[i], rev_conv_dims[i+1], kernel_size=2, stride=2, bias=False),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(rev_conv_dims[i+1], rev_conv_dims[i+1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                                         nn.ReLU()))
        modules.append(nn.ConvTranspose2d(rev_conv_dims[-1], rev_conv_dims[-1], kernel_size=2, stride=2, bias=False))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(rev_conv_dims[-1], self.in_channels[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        modules.append(nn.Sigmoid())
        self.rev_conv = nn.Sequential(*modules)

    def forward(self, z, C, H, W):
        z = z.view(-1, C, H, W) 
        z = self.rev_conv(z)

        z = F.interpolate(z, size=(self.in_channels[1], self.in_channels[2]), mode='nearest')
        return z

class EmbeddingNet(nn.Module):
    def __init__(self, base, **embedding_kwargs):
        super().__init__()
        self.base = base
        self.embedding_kwargs = embedding_kwargs

        self.init_model()
        
    def init_model(self):
        if self.base == 'cnn':
            self.cnn = cnnBlock(self.embedding_kwargs['in_channels'],
                                self.embedding_kwargs['conv_dims'])
            self.embed_dim = self.cnn._calculate_embed_dim()
            self.revcnn = RevcnnBlock(self.embedding_kwargs['conv_dims'][::-1],
                                      self.embedding_kwargs['in_channels'])
        elif self.base == 'resnet50':
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.resnet50.fc = nn.Identity()
            self._freeze_bn(self.resnet50)
            self.embed_dim = self._get_embed_dim(self.resnet50, input_shape=(3, 224, 224))
        elif self.base == 'alexnet':
            self.alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self.alexnet.classifier = nn.Identity()
            self._freeze_bn(self.alexnet)
            self.embed_dim = self._get_embed_dim(self.alexnet, input_shape=(3, 224, 224))
        elif self.base == 'resnet18':
            self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.resnet18.fc = nn.Identity()
            self._freeze_bn(self.resnet18)
            self.embed_dim = self._get_embed_dim(self.resnet18, input_shape=(3, 224, 224))
        else:
            raise ValueError(f'Unknown base model: {self.base}')
        
    def forward(self, x, reverse=False):
        if self.base == 'cnn':
            if not reverse:
                x, (self.C, self.H, self.W) = self.cnn(x)
                return x
            else:
                return self.revcnn(x, self.C, self.H, self.W)
        
        elif self.base == 'resnet50':
            return self.resnet50(x)
        
        elif self.base == 'alexnet':
            return self.alexnet(x)
        
        elif self.base == 'resnet18':
            return self.resnet18(x)
        
        else:
            raise ValueError(f'Unknown base model: {self.base}')
    
    def _freeze_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    @torch.no_grad()
    def _get_embed_dim(self, model, input_shape):
        dummy = torch.zeros(1, *input_shape)
        dummy_out = model(dummy)
        embed_dim = dummy_out.shape[1]
        return embed_dim