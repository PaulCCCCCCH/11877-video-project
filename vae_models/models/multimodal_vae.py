import torch
from torch import nn
from .vae_3d import VideoEncoder, VideoDecoder, Swish

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).
    We will use this for every q(z|x_i) for all i.
    @param n_latents: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """
    def __init__(self, n_latents, n_channels=3):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).
    We will use this for every p(x_i|z) for all i.
    @param n_latents: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """
    def __init__(self, n_latents, n_channels=3):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 5 * 5),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, n_channels, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        return z  # no sigmoid!

      
class MVAE(nn.Module):
    def __init__(self, n_latents=512,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type:str = 'H'):
        super(MVAE, self).__init__()
        
        self.encoders = nn.ModuleList([
          VideoEncoder(n_latents, 1),
          ImageEncoder(n_latents, 3)
        ])
        self.decoders = nn.ModuleList([
          nn.Sequential(VideoDecoder(n_latents, 1), nn.Tanh()),
          ImageDecoder(n_latents, 3)
        ])
        
        self.losses = nn.ModuleList([
          nn.BCEWithLogitsLoss(reduction='sum'),
          nn.BCEWithLogitsLoss(reduction='sum')
        ])
        
        self.recon_loss_factors = [0.3, 1.0]
        
        self.experts = ProductOfExperts()
        self.n_latents = n_latents
        self.num_iter = 0
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = max_capacity
        self.C_stop_iter = capacity_max_iter

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            
            z = eps.mul(std).add_(mu)
            
            # store requires params for training
            self.current_mu = mu
            self.current_log_var = logvar
            
            return z
        else:  # return mean during inference
            return mu
          
    def encode(self, *data_list):
        mu_list, logvar_list = list(), list()
        for data, encoder in zip(data_list, self.encoders):
          mu, logvar = encoder(data)
          mu_list.append(mu)
          logvar_list.append(logvar)
        
        # product of expert
        mu, logvar = self.experts(torch.stack(mu_list), torch.stack(logvar_list))
      
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        
        return z
      
    def forward(self, *data_list):
        z = self.encode(*data_list)
        
        # reconstruct inputs based on sample
        recons = list()
        for decoder in self.decoders:
          recons.append(decoder(z))
       
        return recons
    
    def loss(self, data_in_list, data_recon_list, kld_weight=1.0):
        # kld_weight: Account for the minibatch samples from the dataset
        self.num_iter += 1
        mu = self.current_mu
        log_var = self.current_log_var
        
        batch_size = mu.shape[0]
        recons_loss = 0.0
        
        for data_in, data_recon, loss_func, factor in zip(data_in_list,
                                                          data_recon_list, self.losses, self.recon_loss_factors):
          recons_loss += loss_func(data_recon, data_in) / batch_size * factor
          
        recons_loss /= len(self.losses) # mean over the number of modalities
  
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = min(max(self.C_max/self.C_stop_iter * self.num_iter, 0), self.C_max)
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.item(), 'KLD':kld_loss.item()}
      
      
class MVAEShared(nn.Module):
    def __init__(self, n_latents=512,
                 n_shared_latents=256,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type:str = 'H'):
        super().__init__()
        
        assert n_shared_latents <= n_latents
        
        self.encoders = nn.ModuleList([
          VideoEncoder(n_latents, 3),
          ImageEncoder(n_latents, 3)
        ])
        self.decoders = nn.ModuleList([
          nn.Sequential(VideoDecoder(n_latents, 3), nn.Tanh()),
          ImageDecoder(n_latents, 3)
        ])
        
        self.losses = nn.ModuleList([
          nn.BCEWithLogitsLoss(reduction='sum'),
          nn.BCEWithLogitsLoss(reduction='sum')
        ])
        
        self.recon_loss_factors = [1.0, 1.0]
        
        self.experts = ProductOfExperts()
        self.n_latents = n_latents
        self.n_shared_latent = n_shared_latents
        self.num_iter = 0
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = max_capacity
        self.C_stop_iter = capacity_max_iter

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            
            z = eps.mul(std).add_(mu)
            return z
        else:  # return mean during inference
            return mu
          
    def encode(self, *data_list):
        private_mu_list, private_logvar_list = list(), list()
        shared_mu_list, shared_logvar_list = list(), list()
        for data, encoder in zip(data_list, self.encoders):
          mu, logvar = encoder(data)
          # first n latents are shared
          shared_mu_list.append(mu[:, :self.n_shared_latent])
          shared_logvar_list.append(logvar[:, :self.n_shared_latent])
          # the rest are private
          private_mu_list.append(mu[:, self.n_shared_latent:])
          private_logvar_list.append(logvar[:, self.n_shared_latent:])
        
        # product of expert
        shared_mu, shared_logvar = self.experts(torch.stack(shared_mu_list), torch.stack(shared_logvar_list))
      
        # reparametrization trick to sample
        shared_z = self.reparametrize(shared_mu, shared_logvar)
        
        self.current_shared_mu = shared_mu
        self.current_shared_log_var = shared_logvar
        self.current_private_mu = torch.stack(private_mu_list)
        self.current_private_log_var = torch.stack(private_logvar_list)
        
        self.private_z = self.reparametrize(self.current_private_mu, self.current_private_log_var)
        
        return shared_z
      
    def forward(self, *data_list):
        shared_z = self.encode(*data_list)
        
        # reconstruct inputs based on sample
        recons = list()
        for decoder, private_z in zip(self.decoders, self.private_z):
          recons.append(decoder(torch.cat([shared_z, private_z], dim=1)))
       
        return recons
    
    def loss(self, data_in_list, data_recon_list, kld_weight=1.0, private_loss_weight=1.0):
        # kld_weight: Account for the minibatch samples from the dataset
        self.num_iter += 1

        batch_size = self.current_shared_mu.shape[0]
        recons_loss = 0.0
        
        for data_in, data_recon, loss_func, factor in zip(data_in_list,
                                                          data_recon_list, self.losses, self.recon_loss_factors):
          recons_loss += loss_func(data_recon, data_in) / batch_size * factor
          
        recons_loss /= len(self.losses) # mean over the number of modalities
  
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.current_shared_log_var
                                               - self.current_shared_mu ** 2 -
                                               self.current_shared_log_var.exp(), dim = 1), dim = 0)
    
        kld_loss += (-0.5 * torch.sum(1 + self.current_private_log_var
                                               - self.current_private_mu ** 2 -
                                               self.current_private_log_var.exp(), dim = 2)).mean()
        
        # L1 loss
        private_z_loss = self.private_z.abs().sum() / batch_size / len(self.losses)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss + private_z_loss * private_loss_weight
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = min(max(self.C_max/self.C_stop_iter * self.num_iter, 0), self.C_max)
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.item(),
                'KLD':kld_loss.item(), 'private Z loss': private_z_loss.item()}