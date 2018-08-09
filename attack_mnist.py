import time
import argparse
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.distributions import Normal

from models import VAE, MnistClassifier
from loaders.mnist_single_class import get_mnist_loader


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
  

  def full_sweep(self, cls):
    AS_target = defaultdict(int)
    AS_ignore_target = defaultdict(int)

    classes = list(range(10))
    for src_class in classes:
      for tar_class in classes:
        if src_class == tar_class: continue
        pair = src_class, tar_class
      
        src_loader = get_mnist_loader(src_class, self.args.batch_size)
        tar_loader = get_mnist_loader(tar_class, self.args.batch_size)

        nb_total = 0
        batches = min(len(tar_loader), len(src_loader))
        for src, tar in tqdm(zip(src_loader, tar_loader), total=batches):
          cur_size = min(len(src), len(tar))
          
          src = src[:cur_size].to(self.device)
          tar = tar[:cur_size].to(self.device)
  
          # only take images whose reconstruction is correct into account
          src_recon = self.vae(src)[0]
          src_recon_class = cls(src_recon).argmax(dim=1)
          idx = src_recon_class == src_class

          adv = self._latent_attack(src, tar)[idx]
          assert len(idx.nonzero())  # everything classified incorrectly, weird!
          adv_recon = self.vae(adv)[0]
          adv_recon_class =  cls(adv_recon).argmax(dim=1)
          
          AS_target[pair] += (adv_recon_class == tar_class).sum().item()
          AS_ignore_target[pair] += (adv_recon_class != src_class).sum().item()
      
          nb_total += len(adv)
      
        pair_info = "{}-{}. AS_target: {:.3f} ({}/{}), AS_ignore_target: {:.3f} ({}/{})" \
          .format(src_class, tar_class, AS_target[pair] / nb_total, \
            AS_target[pair], nb_total, AS_ignore_target[pair] / nb_total, \
            AS_ignore_target[pair], nb_total)
        print(pair_info)
        with open(self.args.info_path, 'a') as f:
          f.write(pair_info + '\n')

  def vis(self, src_class, tar_class):
    src_loader = get_mnist_loader(src_class, 1)
    tar_loader = get_mnist_loader(tar_class, 1)
    
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
    tar_recon, _, tar_z_params = self.vae(tar)
    tar_z_params = VAE.flatten_dists_params(tar_z_params).detach()
        
    noise = Normal(loc=src, scale=self.args.noise_std)
    adv = torch.clamp(src + noise.sample(), 0, 1).requires_grad_()
    optimizer = optim.Adam([adv], lr=self.args.eta)
    
    for i in range(self.args.steps):
      adv_recon, _, adv_z_params = self.vae(adv)
      adv_z_params = VAE.flatten_dists_params(adv_z_params)
      
      l2pix = l2_norm(adv - src)
      l2latent = l2_norm(adv_z_params - tar_z_params)
      
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
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--lambda', type=float, default=20, dest='_lambda')
  parser.add_argument('--noise-std', type=float, default=0.2)
  parser.add_argument('--steps', type=int, default=1000,
                      help='number of steps the Adam optimizer takes when \
                            generating an adversarial example')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--info-path', type=str, default='adversarial-info.txt')
  parser.add_argument('--vis-pair', type=str, default='',
                      help='a pair of digit classes to visualize the adversarial \
                            examples of')
  return parser.parse_args()


def run(args):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  # load vae ckpt
  vae_ckpt = torch.load(args.vae_ckpt_path)
  vae = VAE(vae_ckpt['in_dim'], vae_ckpt['cont_dim'], 
    vae_ckpt['cat_dims'], vae_ckpt['temp']).to(device)
  vae.load_state_dict(vae_ckpt['state_dict'])
  vae.eval()
  
  attacker = Attacker(vae, args, device)
  if args.vis_pair:
    try:
      src, tar = map(lambda x: int(x), args.vis_pair.split(','))
    except: 
      raise ValueError("Argument 'vis-pair' needs to be a pair of digits")
    attacker.vis(src, tar)
  else:
    if not args.cls_ckpt_path:
      raise ValueError("Must specify classifier path when evaluating \
                        attacks quantitatviely!")
    # load classifier
    cls = MnistClassifier().to(device)
    state_dict = torch.load(args.cls_ckpt_path)
    cls.load_state_dict(state_dict)

    attacker.full_sweep(cls)


if __name__ == '__main__':
  args = parse()
  run(args)

