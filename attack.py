import argparse
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.distributions import Normal

from models import VAE
from attack_utils.mnist_single_digit import get_mnist_loader


def latent_attack(model, src, tar, device, vis=False):
  src = src.to(device)
  src_recon, _, src_z_params = model(src)
  src_z_params = VAE.flatten_dists_params(src_z_params).detach()
      
  tar = tar.to(device)
  noise = Normal(loc=tar, scale=0.5)
  adv = torch.clamp(tar + noise.sample(), 0, 1).requires_grad_()
  optimizer = optim.Adam([adv], lr=1e-3)
  
  _lambda = 0.5
  runloss, runl2pix = None, None
  for i in range(1000):
    adv_recon, _, adv_z_params = model(adv)
    adv_z_params = VAE.flatten_dists_params(adv_z_params)
    
    l2pix = torch.norm(adv - src)
    l2latent = torch.norm(adv_z_params - src_z_params)
    
    loss = _lambda*l2pix + l2latent
    runloss = loss if not runloss else runloss*0.99 + loss.item()*0.01
    runl2pix = l2pix if not runl2pix else runl2pix*0.99 + l2pix.item()*0.01
    loss.backward()
    optimizer.step()

    adv.data = torch.clamp(adv.data, 0, 1)

    if not i % 100: 
      print("{}: total: {:.3f}, l2 pixel: {:.3f} (with coeff: {:.3f})".format(
        i, runloss, runl2pix.item(), _lambda*runl2pix))

  if vis:
    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(src.squeeze().data, cmap='gray')
    axarr[0, 1].imshow(tar.squeeze().data, cmap='gray')
    axarr[0, 2].imshow(adv.squeeze().data, cmap='gray')

    axarr[1, 0].imshow(src_recon.squeeze().data, cmap='gray')
    tar_recon = model(tar)[0].squeeze().data
    axarr[1, 1].imshow(tar_recon, cmap='gray')
    adv_recon = model(adv)[0].squeeze().data
    axarr[1, 2].imshow(adv_recon, cmap='gray')
    plt.show()


def full_sweep(args, model, device):
  digits = list(range(10))
  for src_digit in digits:
    for tar_digit in digits:
      if src_digit == tar_digit: continue
    
      src_loader = get_mnist_loader(src_digit)
      tar_loader = get_mnist_loader(tar_digit)
      for src, tar in zip(src_loader, tar_loader):
        latent_attack(model, src, tar, device, args.vis) 


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', type=str, required=True, dest='ckpt_path',
                      help='path to the saved VAE model checkpoint')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--vis', action='store_true', default=True)
  return parser.parse_args()


def run(args):
  args = parse()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  # load ckpt
  ckpt = torch.load(args.ckpt_path)
  model = VAE(ckpt['in_dim'], ckpt['cont_dim'], 
    ckpt['cat_dims'], ckpt['temp']).to(device)
  model.load_state_dict(ckpt['state_dict'])
    
  full_sweep(args, model, device)


if __name__ == '__main__':
  args = parse()
  run(args)

