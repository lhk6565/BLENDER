import numpy as np
from tqdm import tqdm
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=256, base='resnet50'):
        super(BaseModel, self).__init__()
        self.base_name = base

        if self.base_name == 'alexnet':
            self.base = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self.base.classifier[6] = nn.Linear(self.base.classifier[6].in_features, hidden_dim)
        elif self.base_name == 'resnet50':
            self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)
        elif self.base_name == 'resnet18':
            self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)
        elif self.base_name == 'cnn':
            self.encoder = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                                         nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),)
            self.fc11 = nn.Sequential(nn.Linear(1024, 64))

            torch.nn.init.xavier_uniform_(self.encoder[0].weight)
            torch.nn.init.xavier_uniform_(self.encoder[4].weight)
            torch.nn.init.xavier_uniform_(self.fc11[0].weight)
            self.fc11[0].bias.data.zero_()

            self.cls = nn.Linear(64, 10)
            torch.nn.init.xavier_uniform_(self.cls.weight)
            self.cls.bias.data.zero_()


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, img_channels=3):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(img_channels+2*c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, img_channels, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c_org, c_trg):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c_org = c_org.view(c_org.size(0), c_org.size(1), 1, 1)
        c_org = c_org.repeat(1, 1, x.size(2), x.size(3))
        c_trg = c_trg.view(c_trg.size(0), c_trg.size(1), 1, 1)
        c_trg = c_trg.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c_org, c_trg], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, img_channels=3):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(img_channels, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2


        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class StarGANTrainer():
    """Solver for training and testing StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""
        config, train_config = config

        # Model configurations.
        self.c_dim = config['domain_dim']
        self.image_size = config['image_shape'][1]
        self.image_channels = config['image_shape'][0]
        self.g_conv_dim = config['embedding_kwargs']['g_conv_dim']
        self.d_conv_dim = config['embedding_kwargs']['d_conv_dim']
        self.g_repeat_num = config['embedding_kwargs']['g_repeat_num']
        self.d_repeat_num = config['embedding_kwargs']['d_repeat_num']
        self.lambda_cls = config['embedding_kwargs']['lambda_cls']
        self.lambda_rec = config['embedding_kwargs']['lambda_rec']
        self.lambda_gp = config['embedding_kwargs']['lambda_gp']

        # Training configurations.
        self.epochs = train_config['epochs']
        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.n_critic = 5
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.patience = train_config['patience']

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.test_domain = config['test_domain']
        self.model_path = config['pretrained_stargan_path']

        # Step size.
        self.decay_start = int(0.5 * self.epochs)
        self.decay_denom = max(1, self.epochs - self.decay_start)

        # Build the model
        self.build_model()
    
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num, self.image_channels).to(self.device)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, self.image_channels).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, betas=(self.beta1, self.beta2))

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, self._lr_lambda)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, self._lr_lambda)

    def _lr_lambda(self, epoch: int):
        if epoch < self.decay_start:
            return 1.0
        # linear decay to 0
        return max(0.0, 1.0 - (epoch - self.decay_start) / (self.decay_denom))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones_like(y, device=self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1) + 1e-12)
        return torch.mean((dydx_l2norm-1)**2)

    def train(self, train_loader, val_loader):
        # Learning rate cache for decaying.
        global_step = 0
        best_val = float("inf")
        counter = 0

        for epoch in tqdm(range(self.epochs), desc='Epoch', leave=False):
            for (x_real, _, label_org) in tqdm(train_loader, desc='starGAN Batch', leave=False):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                x_real = x_real.to(self.device)
                label_org = label_org.to(self.device)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0), device=label_org.device)
                label_trg = label_org[rand_idx]

                c_org = F.one_hot(label_org.long(), num_classes=self.c_dim).float()
                c_trg = F.one_hot(label_trg.long(), num_classes=self.c_dim).float()

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = F.cross_entropy(out_cls, label_org)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_org, c_trg)
                out_src, out_cls = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                if (global_step+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_org, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = F.cross_entropy(out_cls, label_trg)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_trg, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                global_step += 1
            
            self.g_scheduler.step()
            self.d_scheduler.step()


            val_loss = self.evaluate(val_loader)
            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                torch.save(self.G.state_dict(), self.model_path)
                tqdm.write(f'Saved best model at epoch {epoch+1} with best loss {best_val:.4f}')
            else:
                counter += 1
                if counter >= self.patience:
                    tqdm.write(f'Early stopping at epoch {epoch+1}, best loss {best_val:.4f}.')
                    break
                    
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.G.eval()
        losses = []

        for batch_idx, (x_real, _, label_org) in enumerate(val_loader):
            x_real = x_real.to(self.device)
            label_org = label_org.to(self.device)

            rand_idx = torch.randperm(label_org.size(0), device=label_org.device)
            label_trg = label_org[rand_idx]

            c_org = F.one_hot(label_org.long(), num_classes=self.c_dim).float()
            c_trg = F.one_hot(label_trg.long(), num_classes=self.c_dim).float()

            x_fake = self.G(x_real, c_org, c_trg)
            x_reconst = self.G(x_fake, c_trg, c_org)

            rec = torch.mean(torch.abs(x_real - x_reconst))
            losses.append(rec.item())

        self.G.train()
        return float(sum(losses) / max(1, len(losses)))



class DIRT(BaseModel):
    def __init__(self, config):
        super(DIRT, self).__init__(in_channel=config['image_shape'][0], hidden_dim=config['hidden_dim'], base=config['base'])

        self.c_dim = config['domain_dim']
        self.image_channels = config['image_shape'][0]
        self.g_conv_dim = config['embedding_kwargs']['g_conv_dim']
        self.g_repeat_num = config['embedding_kwargs']['g_repeat_num']

        self.stargan = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num, self.image_channels)

        self.out_layer = nn.Linear(config['hidden_dim'], config['num_classes'])
        load_model(self.stargan, config['pretrained_stargan_path'])

        self.stargan.eval()
        self.alpha = 0.01

    def forward(self, x):
        if self.base_name == 'cnn':
            h = self.encoder(x)
            h = h.view(-1, 1024)
            z = self.fc11(h)
            logits = self.cls(F.relu(z))
            return logits, z
        else:
            z = F.relu(self.base(x))
            logits = self.out_layer(z)
            return logits, z
        
    def loss_function(self, x, y, d=None):
        logits, z = self.forward(x)

        loss_cls = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()

        reg = loss_cls.new_zeros(())
        if self.training:
            assert d is not None, "Need domain labels d for reg term during training"

            # domain shuffle
            rand_idx = torch.randperm(d.size(0), device=d.device)
            d_new = d[rand_idx]

            x_new = self.translate(x, d, d_new)
            _, z_new = self.forward(x_new)

            if self.base_name == 'cnn':
                reg = F.mse_loss(z_new, z)
            else:
                reg = self.alpha * F.mse_loss(z_new, z)
    
        loss = loss_cls + reg

        return loss, loss_cls, reg, acc
    
    @torch.no_grad()
    def translate(self, x, d_src, d_trg):
        # d_src, d_trg: (B,) int64, 0..c_dim-1
        c_src = F.one_hot(d_src.long(), num_classes=self.c_dim).float()
        c_trg = F.one_hot(d_trg.long(), num_classes=self.c_dim).float()
        return self.stargan(x, c_src, c_trg)