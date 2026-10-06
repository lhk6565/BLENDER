import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from methods.embedding import EmbeddingNet

### Follows model as seen in LEARNING ROBUST REPRESENTATIONS BY PROJECTING SUPERFICIAL STATISTICS OUT

# Decoders
class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, embed_base='cnn', embed_dim=None, in_channels=1):
        super(px, self).__init__()

        self.embed_base = embed_base

        if self.embed_base == 'cnn':
            self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
            self.up1 = nn.Upsample(scale_factor=2)
            self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU())
            self.up2 = nn.Upsample(scale_factor=2)
            self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False), nn.BatchNorm2d(256), nn.ReLU())
            self.de3 = nn.Sequential(nn.Conv2d(256, in_channels*256, kernel_size=1, stride=1))

            torch.nn.init.xavier_uniform_(self.fc1[0].weight)
            torch.nn.init.xavier_uniform_(self.de1[0].weight)
            torch.nn.init.xavier_uniform_(self.de2[0].weight)
            torch.nn.init.xavier_uniform_(self.de3[0].weight)
            self.de3[0].bias.data.zero_()
        else:
            self.feat_decoder = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 1024, bias=False),
                                              nn.BatchNorm1d(1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, embed_dim),)
            torch.nn.init.xavier_uniform_(self.feat_decoder[0].weight)
            torch.nn.init.xavier_uniform_(self.feat_decoder[3].weight)
            self.feat_decoder[3].bias.data.zero_()

    def forward(self, zd, zx, zy):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)

        if self.embed_base == 'cnn':
            h = self.fc1(zdzxzy)
            h = h.reshape(-1, 64, 4, 4)
            h = self.up1(h)
            h = self.de1(h)
            h = self.up2(h)
            h = self.de2(h)
            loc_img = self.de3(h)
            return loc_img
        else:
            feat = self.feat_decoder(zdzxzy)
            return feat


class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7

        return zd_loc, zd_scale


class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


# Encoders
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, embed_base='cnn', embed_dim=None, in_channels=1):
        super(qzd, self).__init__()

        self.embed_base = embed_base

        if self.embed_base == 'cnn':
            self.encoder = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                                        nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),)

            self.fc11 = nn.Sequential(nn.Linear(1024, zd_dim))
            self.fc12 = nn.Sequential(nn.Linear(1024, zd_dim), nn.Softplus())

            torch.nn.init.xavier_uniform_(self.encoder[0].weight)
            torch.nn.init.xavier_uniform_(self.encoder[4].weight)
            torch.nn.init.xavier_uniform_(self.fc11[0].weight)
            self.fc11[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.fc12[0].weight)
            self.fc12[0].bias.data.zero_()
        else:
            self.mlp = nn.Sequential(nn.Linear(embed_dim, 1024, bias=False),
                                     nn.BatchNorm1d(1024),
                                     nn.ReLU(),)
            self.fc11 = nn.Sequential(nn.Linear(1024, zd_dim))
            self.fc12 = nn.Sequential(nn.Linear(1024, zd_dim), nn.Softplus())

            torch.nn.init.xavier_uniform_(self.mlp[0].weight)
            torch.nn.init.xavier_uniform_(self.fc11[0].weight)
            self.fc11[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.fc12[0].weight)
            self.fc12[0].bias.data.zero_()


    def forward(self, x):
        if self.embed_base == "cnn":
            # x: [B, C, H, W]
            h = self.encoder(x)
            h = h.reshape(h.size(0), -1)  # 원래 1024 가정 유지
        else:
            # x: [B, embed_dim]
            h = self.mlp(x)

        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7

        return zd_loc, zd_scale


class qzx(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, embed_base='cnn', embed_dim=None, in_channels=1):
        super(qzx, self).__init__()

        self.embed_base = embed_base

        if self.embed_base == 'cnn':
            self.encoder = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                                        nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),)

            self.fc11 = nn.Sequential(nn.Linear(1024, zx_dim))
            self.fc12 = nn.Sequential(nn.Linear(1024, zx_dim), nn.Softplus())

            torch.nn.init.xavier_uniform_(self.encoder[0].weight)
            torch.nn.init.xavier_uniform_(self.encoder[4].weight)
            torch.nn.init.xavier_uniform_(self.fc11[0].weight)
            self.fc11[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.fc12[0].weight)
            self.fc12[0].bias.data.zero_()
        else:
            self.mlp = nn.Sequential(nn.Linear(embed_dim, 1024, bias=False),
                                     nn.BatchNorm1d(1024),
                                     nn.ReLU(),)
            
            self.fc11 = nn.Sequential(nn.Linear(1024, zx_dim))
            self.fc12 = nn.Sequential(nn.Linear(1024, zx_dim), nn.Softplus())

            torch.nn.init.xavier_uniform_(self.mlp[0].weight)
            torch.nn.init.xavier_uniform_(self.fc11[0].weight)
            self.fc11[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.fc12[0].weight)
            self.fc12[0].bias.data.zero_()

    def forward(self, x):
        if self.embed_base == 'cnn':
            h = self.encoder(x)
            h = h.reshape(h.size(0), -1)  # [B, 1024] (MNIST 28x28 기준)
        else:
            # x: [B, embed_dim]
            h = self.mlp(x)            # [B, 1024]
        
        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7

        return zx_loc, zx_scale


class qzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, embed_base='cnn', embed_dim=None, in_channels=1):
        super(qzy, self).__init__()

        self.embed_base = embed_base

        if self.embed_base == 'cnn':
            self.encoder = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                                        nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),)

            self.fc11 = nn.Sequential(nn.Linear(1024, zy_dim))
            self.fc12 = nn.Sequential(nn.Linear(1024, zy_dim), nn.Softplus())

            torch.nn.init.xavier_uniform_(self.encoder[0].weight)
            torch.nn.init.xavier_uniform_(self.encoder[4].weight)
            torch.nn.init.xavier_uniform_(self.fc11[0].weight)
            self.fc11[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.fc12[0].weight)
            self.fc12[0].bias.data.zero_()
        else:
            self.mlp = nn.Sequential(nn.Linear(embed_dim, 1024, bias=False),
                                     nn.BatchNorm1d(1024),
                                     nn.ReLU(),)
            
            self.fc11 = nn.Sequential(nn.Linear(1024, zy_dim))
            self.fc12 = nn.Sequential(nn.Linear(1024, zy_dim), nn.Softplus())

            torch.nn.init.xavier_uniform_(self.mlp[0].weight)
            torch.nn.init.xavier_uniform_(self.fc11[0].weight)
            self.fc11[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.fc12[0].weight)
            self.fc12[0].bias.data.zero_()

    def forward(self, x):
        if self.embed_base == 'cnn':
            h = self.encoder(x)
            h = h.reshape(h.size(0), -1)
        else:
            h = self.mlp(x)

        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-7

        return zy_loc, zy_scale


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y


class DIVA(nn.Module):
    def __init__(self, config):
        super(DIVA, self).__init__()
        self.zd_dim = config['zd_dim']
        self.zx_dim = config['zx_dim']
        self.zy_dim = config['zy_dim']
        self.d_dim = config['d_dim']
        self.x_dim = config['x_dim']
        self.y_dim = config['y_dim']
        self.in_channels = config['in_channels']

        self.embed_base = config['embed_base']

        if self.embed_base == 'cnn':
            self.embedding_net = None
            self.embed_dim = None
        else:
            self.embedding_net = EmbeddingNet(base=self.embed_base)
            self.embed_dim = self.embedding_net.embed_dim

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, self.embed_base, self.embed_dim, self.in_channels)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, self.embed_base, self.embed_dim, self.in_channels)
        if self.zx_dim != 0:
            self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, self.embed_base, self.embed_dim, self.in_channels)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, self.embed_base, self.embed_dim, self.in_channels)

        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = config['aux_loss_multiplier_y']
        self.aux_loss_multiplier_d = config['aux_loss_multiplier_d']

        self.beta_x = 1.0
        self.beta_y = 1.0
        self.beta_d = 1.0

    def _q_input(self, x):
        if self.embed_base == 'cnn':
            return x
        self.embedding_net.eval()
        with torch.no_grad():  # backbone freeze
            return self.embedding_net(x)  # [B, embed_dim]
    
    def encode(self, x):
        # Encode
        qx = self._q_input(x)

        zd_q_loc, zd_q_scale = self.qzd(qx)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(qx)
        zy_q_loc, zy_q_scale = self.qzy(qx)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        return qzd.mean, qzx.mean, qzy.mean
    
    def reconstruct(self, x):
        qzd, qzx, qzy = self.encode(x)

        # Decode
        x_recon = self.px(qzd, qzx, qzy)

        x_recon_zd = self.px(qzd, torch.zeros_like(qzx), torch.zeros_like(qzy))
        x_recon_zx = self.px(torch.zeros_like(qzd), qzx, torch.zeros_like(qzy))
        x_recon_zy = self.px(torch.zeros_like(qzd), torch.zeros_like(qzx), qzy)

        return x_recon, (x_recon_zd, x_recon_zx, x_recon_zy)
    
    def pred_y(self, x):
        qx = self._q_input(x)

        zy_q_loc, zy_q_scale = self.qzy(qx)

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.mean

        y_hat = self.qy(zy_q)
        return y_hat
        
    def forward(self, d, x, y):
        # Encode
        qx = self._q_input(x)

        zd_q_loc, zd_q_scale = self.qzd(qx)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(qx)
        zy_q_loc, zy_q_scale = self.qzy(qx)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q)

        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.zx_dim != 0:
            zx_p_loc = torch.zeros_like(zx_q)
            zx_p_scale = torch.ones_like(zx_q)
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y=None):
        if y is None:  # unsupervised
            # Do standard forward pass for everything not involving y
            qx = self._q_input(x)
            zd_q_loc, zd_q_scale = self.qzd(qx)
            if self.zx_dim != 0:
                zx_q_loc, zx_q_scale = self.qzx(qx)
            zy_q_loc, zy_q_scale = self.qzy(qx)

            qzd = dist.Normal(zd_q_loc, zd_q_scale)
            zd_q = qzd.rsample()
            if self.zx_dim != 0:
                qzx = dist.Normal(zx_q_loc, zx_q_scale)
                zx_q = qzx.rsample()
            else:
                zx_q = None
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            zy_q = qzy.rsample()

            zd_p_loc, zd_p_scale = self.pzd(d)
            if self.zx_dim != 0:
                zx_p_loc = torch.zeros_like(zx_q)
                zx_p_scale = torch.ones_like(zx_q)

            pzd = dist.Normal(zd_p_loc, zd_p_scale)

            if self.zx_dim != 0:
                pzx = dist.Normal(zx_p_loc, zx_p_scale)
            else:
                pzx = None

            d_hat = self.qd(zd_q)

            x_recon = self.px(zd_q, zx_q, zy_q)
            
            if self.embed_base == 'cnn':
                B, C, H, W = x.shape  # C=2 for coloredMNIST, C=1 for rotatedMNIST
                # x_recon: [B, C*256, H, W] -> [B, H, W, C, 256]
                logits = x_recon.permute(0, 2, 3, 1).contiguous()           # [B, H, W, C*256]
                logits = logits.view(B, H, W, C, 256).contiguous()
                logits = logits.view(-1, 256)                               # [(B*H*W*C), 256]

                target = (x * 255).long().view(-1)                          # [(B*H*W*C)]
                CE_x = F.cross_entropy(logits, target, reduction='sum')
            else:
                CE_x = F.mse_loss(x_recon, qx, reduction='sum')

            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')


            # Create labels and repeats of zy_q and qzy
            y_onehot = torch.eye(self.y_dim, device=x.device)
            y_onehot = y_onehot.repeat(x.size(0), 1)

            zy_q = zy_q.repeat(self.y_dim, 1)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat(self.y_dim, 1), zy_q_scale.repeat(self.y_dim, 1)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)

            # Do forward pass for everything involving y
            zy_p_loc, zy_p_scale = self.pzy(y_onehot)

            # Reparameterization trick
            pzy = dist.Normal(zy_p_loc, zy_p_scale)

            # Auxiliary losses
            y_hat = self.qy(zy_q)

            # Marginals
            alpha_y = F.softmax(y_hat, dim=-1)
            qy = dist.OneHotCategorical(alpha_y)
            prob_qy = torch.exp(qy.log_prob(y_onehot))

            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q), dim=-1)

            marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)

            prior_y = torch.tensor(1/self.y_dim, device=x.device)
            prior_y_minus_qy = torch.log(prior_y) - qy.log_prob(y_onehot)
            marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)

            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * marginal_zy_p_minus_zy_q \
                   - marginal_prior_y_minus_qy \
                   + self.aux_loss_multiplier_d * CE_d

        else: # supervised
            qx = self._q_input(x)
            x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)
            
            if self.embed_base == 'cnn':
                x_recon_ = x_recon.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                x_recon_ = x_recon_.reshape(-1, 256)
                x_target = (x.reshape(-1) * 255).long()
                CE_x = F.cross_entropy(x_recon_, x_target, reduction='sum')
            else:
                CE_x = F.mse_loss(x_recon, qx, reduction='sum')

            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0

            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            _, y_target = y.max(dim=1)
            CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y,\
                   CE_y

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            qx = self._q_input(x)
            zd_q_loc, zd_q_scale = self.qzd(qx)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)

            zy_q_loc, zy_q_scale = self.qzy.forward(qx)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)

        return d, y