import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical 

from utils.model_utils import *


def _encoder_fancy(img_dims, nb_out, nonl, out_nonl):
  enc_layers = []
  enc_layers.extend(conv_block(img_dims[0], 128, nonl, norm=True))
  enc_layers.extend(conv_block(128, 256, nonl, norm=True))
  enc_layers.extend(conv_block(256, 512, nonl, norm=True))
  enc_layers.extend(conv_block(512, 1024, nonl, norm=True))
  enc_layers.append(Flatten())
  enc_layers.extend(linear_block(1024*4*4, nb_out, out_nonl))
  return nn.Sequential(*enc_layers)


def _decoder_fancy(img_dims, nb_latents, nonl, out_nonl):
  def fancy_deconv_block(in_c, out_c, nonl, norm=True):
    layers = [nn.UpsamplingNearest2d(scale_factor=2)] 
    layers.append(nn.ReplicationPad2d(1))
    layers.append(nn.Conv2d(in_c, out_c, 3, 1))
    if norm: layers.append(nn.BatchNorm2d(out_c, 1e-3))
    layers.append(nonl)
    return layers

  dec_layers = []
  dec_layers.extend(linear_block(nb_latents, 1024*4*4, nonl))
  dec_layers.append(Reshape(-1, 1024, 4, 4))
  dec_layers.extend(fancy_deconv_block(1024, 512, nonl))
  dec_layers.extend(fancy_deconv_block(512, 256, nonl))
  dec_layers.extend(fancy_deconv_block(256, 128, nonl))
  dec_layers.extend(fancy_deconv_block(128, img_dims[0], out_nonl, norm=False))
  return nn.Sequential(*dec_layers)
  

def _encoder(img_dims, nb_out, nonl, out_nonl):
  enc_layers = []
  enc_layers.extend(conv_block(img_dims[0], 32, nonl))
  enc_layers.extend(conv_block(32, 32, nonl))
  enc_layers.extend(conv_block(32, 64, nonl))
  if img_dims[1:] == (64, 64): 
    enc_layers.extend(conv_block(64, 64, nonl))
  enc_layers.append(Flatten())
  enc_layers.extend(linear_block(64*4*4, 128, nonl))
  enc_layers.extend(linear_block(128, nb_out, out_nonl))
  return nn.Sequential(*enc_layers)


def _decoder(img_dims, nb_latents, nonl, out_nonl):
  dec_layers = []
  dec_layers.extend(linear_block(nb_latents, 128, nonl)),
  dec_layers.extend(linear_block(128, 64*4*4, nonl)),
  dec_layers.append(Reshape(-1, 64, 4, 4)),
  if img_dims[1:] == (64, 64):
    dec_layers.extend(deconv_block(64, 64, nonl)),
  dec_layers.extend(deconv_block(64, 32, nonl)),
  dec_layers.extend(deconv_block(32, 32, nonl)),
  dec_layers.extend(deconv_block(32, img_dims[0], out_nonl))
  return nn.Sequential(*dec_layers)


class VAE(nn.Module):
  def __init__(self, img_dims, cont_dim, cat_dims, temp):
    super(VAE, self).__init__()
    self.in_dim = img_dims

    self.cont_dim = [] if not cont_dim else 2*[cont_dim]
    self.cat_dims = cat_dims
    self.chunk_sizes = self.cont_dim + self.cat_dims
    self.temp = temp

    # self.encoder = _encoder(img_dims, sum(self.chunk_sizes), 
    #   nonl=nn.ReLU(), out_nonl=None)
    # self.decoder = _decoder(img_dims, sum(self.chunk_sizes)-cont_dim,   
    #   nonl=nn.ReLU(), out_nonl=nn.Sigmoid())
    self.encoder = _encoder_fancy(img_dims, sum(self.chunk_sizes), 
      nonl=nn.LeakyReLU(), out_nonl=None)
    self.decoder = _decoder_fancy(img_dims, sum(self.chunk_sizes)-cont_dim,   
      nonl=nn.LeakyReLU(), out_nonl=nn.Sigmoid())
  
  @property
  def device(self):
    return next(self.parameters()).device
    
  def _get_dists_params(self, x):
    """Returns the parameters that the encoder predicts."""
    out = self.encoder(x).split(self.chunk_sizes, dim=1)
    params = {}; i = 0
    if self.cont_dim: i += 2; params['cont'] = out[:i]
    if self.cat_dims: params['cat'] = out[i:]
    return params
  
  @staticmethod
  def flatten_dists_params(params_dict):
    params_flat = []
    if 'cont' in params_dict.keys():
      src_mu = params_dict['cont'][0]
      params_flat.append(src_mu)
    if 'cat' in params_dict.keys():
      for logits in params_dict['cat']:
        params_flat.append(logits)
    params_flat = torch.cat(params_flat, dim=1)
    return params_flat
    
  def _reparam_gauss(self, mu, logvar):
    if self.training:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_().requires_grad_()
        return eps.mul(std).add_(mu)
    else:
        return mu
  
  def sample(self, n):
    """Samples n datapoints from the prior and reconstructs them."""
    z = []
    if len(self.cont_dim):
      d = self.cont_dim[0]
      z_cont = self._reparam_gauss(torch.zeros(n, d), torch.ones(n, d))
      z.append(z_cont)
    for cat_dim in self.cat_dims:
      z_cat = torch.zeros(n, cat_dim)
      indices = Categorical(torch.ones(n, cat_dim) / cat_dim).sample()
      z_cat[torch.arange(n, out=torch.LongTensor()), indices] = 1
      # z_cat = F.gumbel_softmax(torch.ones(n, cat_dim), tau=self.temp) 
      z.append(z_cat)
    z = torch.cat(z, dim=1).to(self.device)
    return self.decoder(z)
  
  def forward(self, x, decode=True):
    params = self._get_dists_params(x)
    zs = []
    if 'cont' in params.keys():
      zs = [self._reparam_gauss(*params['cont'])]
    if 'cat' in params.keys():
      for logits in params['cat']:
        if self.training: zs.append(F.gumbel_softmax(logits, tau=self.temp))
        else: zs.append(F.gumbel_softmax(logits, tau=self.temp, hard=True))
    z = torch.cat(zs, dim=1)
    
    if decode: recon = self.decoder(z)
    else: recon = None
    return recon, z, params


def tc_disc(nb_latents):
  """Discriminator for estimating the mutual information 
     under the aggregate posterior q(z)."""
  lrelu = nn.LeakyReLU()
  return nn.Sequential(
    *linear_block(nb_latents, 1000, nonl=lrelu),
    *linear_block(1000, 1000, nonl=lrelu),
    *linear_block(1000, 1000, nonl=lrelu),
    *linear_block(1000, 1000, nonl=lrelu),
    *linear_block(1000, 1000, nonl=lrelu),
    *linear_block(1000, 1, nonl=None)
  )


class MnistClassifier(nn.Module):
  def __init__(self):
    super(MnistClassifier, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(500, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 500)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

