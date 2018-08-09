import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.distributions import Normal

from models import VAE
from attack_utils.train_cls import MnistClassifier
from attack_utils.mnist_single_loader import get_mnist_loader


class Attacker():
  def __init__(self, vae, args, device):
    self.vae = vae
    self.args = args
    self.device = device
  

  def full_sweep(self, cls):
    AS_target = defaultdict(int)
    AS_ignore_target = defaultdict(int)

    classes = list(range(10))
    for src_class in classes:
      for tar_class in classes:
        if src_class == tar_class: continue
      
        src_loader = get_mnist_loader(src_class)
        tar_loader = get_mnist_loader(tar_class)
        for i, (src, tar) in enumerate(zip(src_loader, tar_loader)):
          if i >= self.args.attacks_per_pair: break
          adv = self._latent_attack(src, tar) 
      
          adv_recon = self.vae(adv)[0]
          return cls(adv_recon).argmax().item()

          if adv_recon_class == tar_class:
            AS_target[src_class, tar_class] += 1
          if adv_recon_class != src_class:
            AS_ignore_target[src_class, tar_class] += 1
      
        print("{}/{}. AS_target: {}, AS_ignore_target: {}" \
          .format(src_class, tar_class, 
            AS_target[src_class, tar_class],
            AS_ignore_target[src_class, tar_class]))
  

  def vis(self, src_class, tar_class):
    src_loader = get_mnist_loader(src_class)
    tar_loader = get_mnist_loader(tar_class)
    
    for src, tar in zip(src_loader, tar_loader):
      src = src.to(self.device)
      tar = tar.to(self.device)

      adv = self._latent_attack(src, tar)

      f, axarr = plt.subplots(2, 3)
      axarr[0, 0].imshow(src.squeeze().data, cmap='gray')
      axarr[0, 1].imshow(tar.squeeze().data, cmap='gray')
      axarr[0, 2].imshow(adv.squeeze().data, cmap='gray')

      src_recon = self.vae(src)[0].squeeze().data
      axarr[1, 0].imshow(src_recon, cmap='gray')
      tar_recon = self.vae(tar)[0].squeeze().data
      axarr[1, 1].imshow(tar_recon, cmap='gray')
      adv_recon = self.vae(adv)[0].squeeze().data
      axarr[1, 2].imshow(adv_recon, cmap='gray')
      plt.show()


  def _latent_attack(self, src, tar):
    tar = tar.to(self.device)
    tar_recon, _, tar_z_params = self.vae(tar)
    tar_z_params = VAE.flatten_dists_params(tar_z_params).detach()
        
    src = src.to(self.device)
    noise = Normal(loc=src, scale=self.args.noise_std)
    adv = torch.clamp(src + noise.sample(), 0, 1).requires_grad_()
    optimizer = optim.Adam([adv], lr=self.args.eta)
    
    for i in range(self.args.steps):
      adv_recon, _, adv_z_params = self.vae(adv)
      adv_z_params = VAE.flatten_dists_params(adv_z_params)
      
      l2pix = torch.norm(adv - src)
      l2latent = torch.norm(adv_z_params - tar_z_params)
      
      loss = self.args._lambda*l2pix + l2latent
      loss.backward()
      optimizer.step()

      adv.data = torch.clamp(adv.data, 0, 1)
        
    return adv
    

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vae-ckpt', type=str, required=True, dest='vae_ckpt_path')
  parser.add_argument('--cls-ckpt', type=str, default=None, dest='cls_ckpt_path')
  parser.add_argument('--eta', type=float, default=1e-3)
  parser.add_argument('--lambda', type=float, default=20, dest='_lambda')
  parser.add_argument('--noise-std', type=float, default=0.2)
  parser.add_argument('--steps', type=int, default=1000,
                      help='number of steps the Adam optimizer takes when \
                            generating an adversarial example')
  parser.add_argument('--attacks-per-pair', type=int, default=10) 
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--vis', action='store_true', default=False)
  return parser.parse_args()


def run(args):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  # load vae ckpt
  vae_ckpt = torch.load(args.vae_ckpt_path)
  vae = VAE(vae_ckpt['in_dim'], vae_ckpt['cont_dim'], 
    vae_ckpt['cat_dims'], vae_ckpt['temp']).to(device)
  vae.load_state_dict(vae_ckpt['state_dict'])
  
  attacker = Attacker(vae, args, device)
  if args.vis:
    attacker.vis(4, 6)
  else:
    if not args.cls_ckpt_path:
      raise ValueError("Must specify classifier path when evaluating
                        attacks quantitatviely!")
    # load classifier
    cls = MnistClassifier().to(device)
    state_dict = torch.load(args.cls_ckpt_path)
    cls.load_state_dict(state_dict)

    attacker.full_sweep()


if __name__ == '__main__':
  args = parse()
  run(args)

