import time
import argparse
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
from torchvision.utils import save_image

from models import VAE
from loaders.celeba import CelebAUtil


def l2_norm(x):
  """Takes the L2 norm of the input tensor across all dimensions 
     except the 0-th, the latter of which is summer across."""
  return x.view(len(x), -1).pow(2).sum(1).sqrt().sum()


class Attacker():
  def __init__(self, vae, args, device):
    self.vae = vae
    self.args = args
    self.device = device
    
    # gradients w.r.t. weights not needed
    for p in self.vae.parameters(): p.requires_grad = False
  
  def vis_bulk(self, n):
    testdata = CelebAUtil().testdata
    srcs, advs, adv_recons = [], [], []
    for i in tqdm(range(n)):
      src = testdata[i].unsqueeze(0).to(self.device)
      tar = testdata[0].unsqueeze(0).to(self.device)
      
      adv = self._latent_attack(src, tar)
      adv_recon = self.vae(adv)[0]
    
      srcs.append(src.squeeze())
      advs.append(adv.squeeze())
      adv_recons.append(adv_recon.squeeze())
      
    srcs = torch.stack(srcs)
    advs = torch.stack(advs)
    adv_recons = torch.stack(adv_recons)

    save_image(srcs, "srcs.png", nrow=int(np.sqrt(n)), pad_value=1)
    save_image(advs, "advs.png", nrow=int(np.sqrt(n)), pad_value=1)
    save_image(adv_recons, "adv_recons.png", nrow=int(np.sqrt(n)), pad_value=1)

  def _latent_attack(self, src, tar):
    tar_recon, _, tar_z_params = self.vae(tar)
    tar_z_params = VAE.flatten_dists_params(tar_z_params).detach()
        
    noise = Normal(loc=src, scale=self.args.noise_std)
    adv = torch.clamp(src + noise.sample(), 0, 1).requires_grad_()
    optimizer = optim.Adam([adv], lr=self.args.eta)
  
    # per-pixel lower and upper bounds
    _min = torch.max(torch.zeros_like(src.data), src.data-self.args.eps)
    _max = torch.min(torch.ones_like(src.data), src.data+self.args.eps)
    
    for i in range(self.args.steps):
      adv_recon, _, adv_z_params = self.vae(adv)
      adv_z_params = VAE.flatten_dists_params(adv_z_params)
      
      l2pix = l2_norm(adv - src)
      l2latent = l2_norm(adv_z_params - tar_z_params)
      
      loss = self.args._lambda*l2pix + l2latent
      loss.backward()
      optimizer.step()

      adv.data = torch.min(torch.max(adv.data, _min), _max)
        
    return adv
    

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', type=str, required=True)
  parser.add_argument('--eta', type=float, default=1e-3)
  parser.add_argument('--lambda', type=float, default=1, dest='_lambda',
                      help='coefficient for the pixel-wise norm')
  parser.add_argument('--eps', type=float, default=0.1,
                      help='the maximum amount any pixel in the adversarial \
                            image can deviate from a corresponding pixel in the \
                            source image')
  parser.add_argument('--noise-std', type=float, default=0.2)
  parser.add_argument('--steps', type=int, default=1000,
                      help='number of steps the Adam optimizer takes when \
                            generating an adversarial example')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--n', type=int, default=4,   
                      help='number of samples to visualize')
  return parser.parse_args()


def run(args):
  torch.manual_seed(args.seed)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  # load vae ckpt
  vae_ckpt = torch.load(args.ckpt)
  vae = VAE(vae_ckpt['in_dim'], vae_ckpt['cont_dim'], 
    vae_ckpt['cat_dims'], vae_ckpt['temp']).to(device)
  vae.load_state_dict(vae_ckpt['state_dict'])
  vae.eval()
  
  attacker = Attacker(vae, args, device)
  assert args.n > 0 and np.sqrt(args.n).is_integer()
  attacker.vis_bulk(args.n)


if __name__ == '__main__':
  args = parse()
  run(args)

