import torch
from torch import nn
from .language_model import get_last_element

def add_loss(loss_dict, loss, name):
  loss_dict[name] = loss.item()
  loss_dict["loss"] += loss


# regular 
class VideoEncoderWithLM(nn.Module):
  def __init__(self, video_encoder, lang_model, lm_proj_mlp):
    super().__init__()
    self.video_encoder = video_encoder
    self.lang_model = lang_model
    self.lm_linear_proj = lm_proj_mlp
    # self.embedding_similarity_loss = nn.CosineEmbeddingLoss(margin=0.1, reduction="sum") # unable to train
    self.embedding_similarity_loss = nn.L1Loss(reduction="sum")
    
  def forward(self, video, text, text_len, train_lm=False):
    run_lm = lambda: get_last_element(self.lang_model(text)[0], text_len)
    
    video_embedding = self.video_encoder.encode(video)
    
    if train_lm:
      text_embedding = run_lm()
    else:
      with torch.no_grad():
        text_embedding = run_lm()
        
    return video_embedding, self.lm_linear_proj(text_embedding)
  
  
  def loss(self, video, video_embedding, text_embedding, weights={"kld":1.0, "sim":1.0}):
    batch_size = video.size(0)
    loss_dict = self.video_encoder.loss(video, self.video_encoder.decode(video_embedding), weights["kld"])
    embedding_similarity_loss = self.embedding_similarity_loss(text_embedding, video_embedding)
                                                               #torch.ones(batch_size, dtype=torch.long).to(video.device))
    add_loss(loss_dict, embedding_similarity_loss / batch_size * weights["sim"],
             "embed_sim")
    
    return loss_dict
  

from .multimodal_vae import ProductOfExperts

class VideoEncoderWithLMExpert(nn.Module):
  def __init__(self, video_encoder, lang_model, lm_proj_mlp):
    super().__init__()
    self.video_encoder = video_encoder
    self.lang_model = lang_model
    self.lm_linear_proj = lm_proj_mlp
    self.expert = ProductOfExperts()

    
  def reparametrize(self, mu, logvar):
      if self.training:
          std = logvar.mul(0.5).exp_()
          eps = std.data.new(std.size()).normal_()
          z = eps.mul(std).add_(mu)

          self.current_mu = mu
          self.current_log_var = logvar

          return z
      else:  # return mean during inference
          return mu
          
  def forward(self, video, text, text_len, train_lm=False):
    run_lm = lambda: get_last_element(self.lang_model(text)[0], text_len)
    
    self.video_encoder.encode(video)
    
    if train_lm:
      text_embedding = run_lm()
    else:
      with torch.no_grad():
        text_embedding = run_lm()
        
    lm_expert = self.lm_linear_proj(text_embedding)
    
    mu, logvar = self.expert(torch.stack([self.video_encoder.current_mu, lm_expert[:, :self.video_encoder.n_latents]]),
                              torch.stack([self.video_encoder.current_log_var, lm_expert[:, self.video_encoder.n_latents:]]))

    # reparametrization trick to sample
    z = self.reparametrize(mu, logvar)
    
    # reuse VAE loss by hacking into the video encoder module
    self.video_encoder.current_mu = mu
    self.video_encoder.current_log_var = logvar
    
    return z
  
  def loss(self, video, embedding, weights={"kld":1.0}):
    return self.video_encoder.loss(video, self.video_encoder.decode(embedding), weights["kld"])
  
  
class VideoEncoderWithLMExpertShared(nn.Module):
  def __init__(self, video_encoder, lang_model, lm_proj_mlp, n_shared):
    super().__init__()
    self.video_encoder = video_encoder
    self.lang_model = lang_model
    self.lm_linear_proj = lm_proj_mlp
    self.expert = ProductOfExperts()
    self.n_shared = n_shared

    
  def reparametrize(self, mu, logvar):
      if self.training:
          std = logvar.mul(0.5).exp_()
          eps = std.data.new(std.size()).normal_()
          z = eps.mul(std).add_(mu)

          self.current_mu = mu
          self.current_log_var = logvar

          return z
      else:  # return mean during inference
          return mu
          
  def forward(self, video, text, text_len, train_lm=False):
    run_lm = lambda: get_last_element(self.lang_model(text)[0], text_len)
    
    self.video_encoder.encode(video)
    
    if train_lm:
      text_embedding = run_lm()
    else:
      with torch.no_grad():
        text_embedding = run_lm()
        
    lm_expert = self.lm_linear_proj(text_embedding)
    
    shared_mu, shared_logvar = self.expert(
      torch.stack([self.video_encoder.current_mu[:, :self.n_shared], lm_expert[:, :self.n_shared]]),
      torch.stack([self.video_encoder.current_log_var[:, :self.n_shared], lm_expert[:, self.n_shared:]])
    )

    # reparametrization trick to sample
    shared_z = self.reparametrize(shared_mu, shared_logvar)
    private_z = self.reparametrize(
      self.video_encoder.current_mu[:, self.n_shared:],
      self.video_encoder.current_log_var[:, self.n_shared:]
    )
    
    # reuse VAE loss by hacking into the video encoder module
    self.video_encoder.current_mu = torch.cat([shared_mu,
                                               self.video_encoder.current_mu[:, self.n_shared:]], dim=1)
    self.video_encoder.current_log_var = torch.cat([shared_logvar,
                                                    self.video_encoder.current_log_var[:, self.n_shared:]], dim=1)
    self.private_z = private_z
    return torch.cat([shared_z, private_z], dim=1)
  
  def loss(self, video, embedding, weights={"kld":1.0, "privZ": 1.0}):
    losses = self.video_encoder.loss(video, self.video_encoder.decode(embedding), weights["kld"])
    add_loss(losses, self.private_z.abs().sum() / video.size(0) * weights["privZ"], "privZ")
    return losses