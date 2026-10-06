import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.embedding import EmbeddingNet
from methods.hsic import *


class InvariantBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.invariant_net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                           nn.BatchNorm1d(output_dim),
                                           nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.invariant_net(x)


class SpecificBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.specific_net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                          nn.BatchNorm1d(output_dim),
                                          nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.specific_net(x)
    

class PredictSpecificDist(nn.Module):
    def __init__(self, d_dim, z_spec_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_dim, z_spec_dim),
                                 nn.BatchNorm1d(z_spec_dim),
                                 nn.ReLU(inplace=True))
        self.pred_mu = nn.Linear(z_spec_dim, z_spec_dim)
        self.logvar = nn.Linear(z_spec_dim, z_spec_dim)

    def forward(self, d):
        hidden = self.net(d)
        pred_spec_mu = self.pred_mu(hidden)
        pred_logvar = self.logvar(hidden)
        return [pred_spec_mu, pred_logvar]
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim))
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class Encoder(nn.Module):
    def __init__(self, input_dim, inv_dims, spec_dims, z_inv_dim, z_spec_dim):
        super().__init__()
        assert len(inv_dims) == len(spec_dims)
        L = len(inv_dims)
        
        self.inv_blocks  = nn.ModuleList()
        self.spec_blocks = nn.ModuleList()
        for i in range(L):
            inv_in  = input_dim if i == 0 else inv_dims[i-1]
            spec_in = input_dim if i == 0 else (inv_dims[i-1] + spec_dims[i-1])
            self.inv_blocks.append(InvariantBlock(inv_in, inv_dims[i]))
            self.spec_blocks.append(SpecificBlock(spec_in, spec_dims[i]))

        self.inv_mu      = nn.Linear(inv_dims[-1], z_inv_dim)
        self.inv_logvar  = nn.Linear(inv_dims[-1], z_inv_dim)

    def forward(self, x):
        h_inv, h_spec = None, None
        for i, (inv_block, spec_block) in enumerate(zip(self.inv_blocks, self.spec_blocks)):
            if i == 0:
                h_inv  = inv_block(x)
                h_spec = spec_block(x)
            else:
                cat_prev = torch.cat([h_inv.detach(), h_spec], dim=1)
                h_inv  = inv_block(h_inv)
                h_spec = spec_block(cat_prev)

        inv_mu     = self.inv_mu(h_inv)
        inv_logvar = self.inv_logvar(h_inv)

        return [inv_mu, inv_logvar], h_spec


class Decoder(nn.Module):
    def __init__(self, latent_inv_dim, latent_spec_dim, dec_dims, output_dim):
        super().__init__()
        layers = []
        latent_dim = latent_inv_dim + latent_spec_dim

        for i in range(len(dec_dims)):
            input_dim = latent_dim if i == 0 else dec_dims[i-1]
            layers.append(nn.Linear(input_dim, dec_dims[i]))
            layers.append(nn.BatchNorm1d(dec_dims[i]))
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)
        self.dense = nn.Linear(dec_dims[-1], output_dim)
        
    def forward(self, z_inv, z_spec):
        x = torch.cat([z_inv, z_spec], dim=1)
        x = self.net(x)
        return self.dense(x)


class ResnetDecoder(nn.Module):
    def __init__(self, latent_inv_dim, latent_spec_dim, dec_dims, output_dim):
        super().__init__()
        layers = []
        latent_dim = latent_inv_dim + latent_spec_dim

        for i in range(len(dec_dims)):
            input_dim = latent_dim if i == 0 else dec_dims[i-1]
            layers.append(nn.Linear(input_dim, dec_dims[i]))
            layers.append(nn.BatchNorm1d(dec_dims[i]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dec_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)
        self.res_block = ResidualBlock(output_dim)
        self.dense = nn.Linear(output_dim, output_dim)
        
    def forward(self, z_inv, z_spec):
        z = torch.cat([z_inv, z_spec], dim=1)
        x = self.res_block(self.net(z))
        return self.dense(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)
 

class Blender(nn.Module):
    def __init__(self, inv_dims, spec_dims, z_inv_dim, z_spec_dim, dec_dims, domain_dim, num_classes, embed_base=None, **kwargs):
        super().__init__()
    
        self.inv_dims = inv_dims
        self.spec_dims = spec_dims
        self.z_inv_dim = z_inv_dim
        self.z_spec_dim = z_spec_dim
        self.dec_dims = dec_dims
        self.domain_dim = domain_dim
        self.num_classes = num_classes
        self.embed_base = embed_base
        self.eps = 1e-8

        embedding_kwargs = kwargs.get('embedding_kwargs')

        if self.embed_base == 'cnn':
            self.embedding_net = EmbeddingNet(base=self.embed_base, **embedding_kwargs)
            self.embed_dim = self.embedding_net.embed_dim
            self.decoder = Decoder(latent_inv_dim=self.z_inv_dim,
                                   latent_spec_dim=self.z_spec_dim, 
                                   dec_dims=self.dec_dims,
                                   output_dim=self.embed_dim)
        else:
            self.embedding_net = EmbeddingNet(base=self.embed_base)
            self.embed_dim = self.embedding_net.embed_dim
            self.decoder = ResnetDecoder(latent_inv_dim=self.z_inv_dim,
                                         latent_spec_dim=self.z_spec_dim,
                                         dec_dims=self.dec_dims,
                                         output_dim=self.embed_dim)

        self.encoder = Encoder(input_dim=self.embed_dim,
                               inv_dims=self.inv_dims,
                               spec_dims=self.spec_dims,
                               z_inv_dim=self.z_inv_dim,
                               z_spec_dim=self.z_spec_dim)
        
        self.classifier = Classifier(input_dim=self.z_inv_dim, num_classes=num_classes)

        self.pred_spec_dist = PredictSpecificDist(d_dim=self.domain_dim,
                                                  z_spec_dim=self.z_spec_dim)
        
        self.spec_hidden = nn.Linear(spec_dims[-1] + z_inv_dim, spec_dims[-1])
        self.spec_mu     = nn.Linear(spec_dims[-1], z_spec_dim)
        self.spec_logvar = nn.Linear(spec_dims[-1], z_spec_dim)

    def _reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def encode(self, x):
        if self.embed_base == 'cnn':
            self.embed_x = self.embedding_net(x, reverse=False)
        else:
            self.embed_x = self.embedding_net(x)

        [self.mu_inv, self.logvar_inv], h_spec = self.encoder(self.embed_x)

        if self.training:
            z_inv = self._reparameterize(self.mu_inv, self.logvar_inv)
        else:
            z_inv = self.mu_inv

        h_spec_in   = torch.cat([z_inv.detach(), h_spec], dim=1)
        h_spec_mid  = torch.relu(self.spec_hidden(h_spec_in))
        self.mu_spec     = self.spec_mu(h_spec_mid)
        self.logvar_spec = self.spec_logvar(h_spec_mid)

        if self.training:
            z_spec = self._reparameterize(self.mu_spec, self.logvar_spec)
        else:
            z_spec = self.mu_spec
        return (z_inv, z_spec)
    
    def decode(self, z_inv, z_spec):
        x_rec = self.decoder(z_inv, z_spec)

        if self.embed_base == 'cnn':
            self.x_rec = self.embedding_net(x_rec, reverse=True)
        else:
            self.x_rec = x_rec

        return self.x_rec
    
    def target_predict(self, z_inv):
        y_pred = self.classifier(z_inv)
        return y_pred

    def forward(self, x):
        self.z_inv, self.z_spec = self.encode(x)
        
        self.x_rec = self.decode(self.z_inv, self.z_spec)

        self.y_pred = self.target_predict(self.z_inv)

        return self.x_rec, self.y_pred, (self.z_inv, self.z_spec)
    
    def loss_function(self, x, y, d, beta_inv, beta_spec, lambda_hsic_dinv, lambda_hsic_inv_spec, sigma_hsic=1.0):
        # Target prediction loss
        task_loss = F.cross_entropy(self.y_pred, y, reduction='mean')
        
        # Reconstruction loss
        if self.embed_base == 'cnn':
            recon_loss = F.binary_cross_entropy(self.x_rec, x, reduction='mean')
        else:
            recon_loss = F.mse_loss(self.x_rec, self.embed_x, reduction='mean')

        # KL divergence losses
        (self.pred_mu_spec, self.pred_logvar_spec) = self.pred_spec_dist(d)

        # Invariant: KL(q(z_inv|x) || N(0,I))
        var_inv = torch.exp(self.logvar_inv)
        kld_inv = 0.5 * (self.mu_inv.pow(2) + var_inv - self.logvar_inv - 1).sum(dim=1).mean()

        # Specific: KL(q(z_spec|x) || p(z_spec|d))
        var_spec = torch.exp(self.logvar_spec)  # σ_q^2
        prior_var_spec = torch.exp(self.pred_logvar_spec)  # σ_p^2

        kld_spec = 0.5 * ((self.pred_logvar_spec - self.logvar_spec)  # log σ_p^2 - log σ_q^2
                          + (var_spec + (self.mu_spec - self.pred_mu_spec).pow(2)) / (prior_var_spec + self.eps) - 1).sum(dim=1).mean()

        # HSIC regularizers (minimize)
        # 1) minimize HSIC(d, z_inv) where d is one-hot
        hsic_d_inv = hsic_biased(x=self.z_inv,
                                 y=d.float(),
                                 sigma_x=sigma_hsic,
                                 k_type_x="gaussian",
                                 k_type_y="delta",
                                 clamp_nonneg=True)
        # 2) minimize HSIC(z_inv, z_spec)
        hsic_inv_spec = hsic_biased(x=self.z_inv,
                                    y=self.z_spec,
                                    sigma_x=sigma_hsic,
                                    k_type_x="gaussian",
                                    k_type_y="gaussian",
                                    clamp_nonneg=True)

        loss = recon_loss + task_loss + beta_inv * kld_inv + beta_spec * kld_spec + lambda_hsic_dinv * hsic_d_inv + lambda_hsic_inv_spec * hsic_inv_spec

        return loss, (recon_loss, task_loss, kld_inv, kld_spec, hsic_d_inv, hsic_inv_spec)
