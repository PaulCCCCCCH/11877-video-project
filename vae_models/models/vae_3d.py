import torch
from torch import nn

class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * x.sigmoid()
      
class VideoEncoder(nn.Module):
    """Parametrizes q(z|x).
    We will use this for every q(z|x_i) for all i.
    @param n_latents: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """
    def __init__(self, n_latents, n_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(n_channels, 32, (3, 4, 4), (2, 2, 2), (1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            Swish(),
            nn.Conv3d(32, 64, (2, 4, 4), 2, (1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            Swish(),
            nn.Conv3d(64, 128, (2, 4, 4), 2, (1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            Swish(),
            nn.Conv3d(128, 256, (2, 4, 4), (1, 1, 1), 0, bias=False),
            nn.BatchNorm3d(256),
            Swish())
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, n_latents * 4),
            Swish(),
            nn.Dropout(p=0.1), 
            nn.Linear(n_latents * 4, n_latents * 2),
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class VideoDecoder(nn.Module):
    """Parametrizes p(x|z).
    We will use this for every p(x_i|z) for all i.
    @param n_latents: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """
    def __init__(self, n_latents, n_channels=3):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 5 * 5),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2, 4, 4), (1, 1, 1), 0, output_padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(128),
            Swish(),
            nn.ConvTranspose3d(128, 64, (2, 4, 4), 2, (1, 1, 1), output_padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            Swish(),
            nn.ConvTranspose3d(64, 32, (2, 4, 4), 2, (1, 1, 1), output_padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(32),
            Swish(),
            nn.ConvTranspose3d(32, n_channels, (3, 4, 4), (2, 2, 2), (1, 1, 1), bias=False)
        )

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 1, 5, 5)
        z = self.hallucinate(z)
        return z  # no sigmoid!

class VAE3D(nn.Module):
    def __init__(self, n_latents=512,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type:str = 'H'):
        super().__init__()
        # define q(z|x_i) for i = 1...6
        self.image_encoder     = VideoEncoder(n_latents, 3)
        # define p(x_i|z) for i = 1...6
        self.image_decoder     = VideoDecoder(n_latents, 3)
        # define q(z|x) = q(z|x_1)...q(z|x_6)
        self.n_latents = n_latents
        self.bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.num_iter = 0
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter

    def reparametrize(self, mu, logvar):
        self.current_mu = mu
        self.current_log_var = logvar
        
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
            return z
        else:  # return mean during inference
            return mu
    
    def encode(self, image):
        mu, logvar = self.image_encoder(image)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        
        return z
      
    def decode(self, z):
        # reconstruct inputs based on sample
        return self.image_decoder(z)
    
    def forward(self, image):
        return self.decode(self.encode(image))
      
    def loss(self, data_in, data_recon, kld_weight=1.0):
        # kld_weight: Account for the minibatch samples from the dataset
        self.num_iter += 1
        recons = data_recon
        input = data_in
        mu = self.current_mu
        log_var = self.current_log_var
        
        # since the image value is normalized between 0~1, BCE loss is better
        batch_size = recons.shape[0]
        recons_loss = self.bce_logit_loss(recons, input) / batch_size# / 10 # a constant to reduce recon influcence
  
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.item(), 'KLD':kld_loss.item()}
