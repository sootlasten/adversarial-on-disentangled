import argparse
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
from torchvision import datasets, transforms

from models import VAE


def latent_attack(args, model, device, dataloader):
  for (x, y), _ in dataloader:
    y = y.unsqueeze(0).to(device)
    recon_y, _, y_z_params = model(y)
    mu_y, logvar_y = y_z_params['cont']
    mu_y = mu_y.detach()
      
    x = x.unsqueeze(0).to(device).detach()
    normal = Normal(loc=x, scale=0.5)
    #uniform = Uniform(torch.zeros_like(x), torch.ones_like(x))
    #adv = Variable(uniform.sample(), requires_grad=True)
    adv = torch.clamp(x + normal.sample(), 0, 1).requires_grad_()
    #plt.imshow(adv.squeeze().data, cmap='gray'); plt.show()
    optimizer = optim.Adam([adv], lr=1e-3)
    
    _lambda = 0.5
    runloss, runl2pix = None, None
    for i in range(1000):
      recon_adv, _, adv_z_params = model(adv)
      mu_adv, _ = adv_z_params['cont']
      
      l2pix = torch.norm(adv - x)
      l2latent = torch.norm(mu_adv - mu_y)
      #l2recon = torch.norm(recon_adv - recon_y.detach())
      #l2recon = torch.norm(recon_adv - y.view(-1).detach())

      loss = _lambda*l2pix + l2latent
      runloss = loss if not runloss else runloss*0.99 + loss.item()*0.01
      runl2pix = l2pix if not runl2pix else runl2pix*0.99 + l2pix.item()*0.01
      loss.backward()
      optimizer.step()

      adv.data = torch.clamp(adv.data, 0, 1)

      if not i % 100: 
        print("{}: total: {:.3f}, l2 pixel: {:.3f} (with coeff: {:.3f})".format(
          i, runloss, runl2pix.item(), _lambda*runl2pix))

    if args.vis:
      recon_x = model(x)[0].squeeze()
      recon_y = model(recon_y)[0].squeeze()
      recon_adv = model(adv)[0].squeeze()
      
      f, axarr = plt.subplots(2, 3)
      axarr[0, 0].imshow(x.data.squeeze(), cmap='gray')
      axarr[0, 1].imshow(y.data.squeeze(), cmap='gray')
      axarr[0, 2].imshow(adv.data.squeeze(), cmap='gray')

      axarr[1, 0].imshow(recon_x.data, cmap='gray')
      axarr[1, 1].imshow(recon_y.data.squeeze(), cmap='gray')
      axarr[1, 2].imshow(recon_adv.data, cmap='gray')
      plt.show()


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', type=str, required=True, dest='ckpt_path',
                      help='path to the saved VAE model checkpoint')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--vis', action='store_true', default=True)
  return parser.parse_args()


def main(args):
  args = parse()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  # load ckpt
  ckpt = torch.load(args.ckpt_path)
  model = VAE(ckpt['in_dim'], ckpt['cont_dim'], ckpt['cat_dims']).to(device)
  model.load_state_dict(ckpt['state_dict'])
  
  # load data
  all_transforms = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor()
  ])
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, 
      transform=all_transforms, download=True),
    batch_size=2, shuffle=False, **kwargs)

  latent_attack(args, model, device, test_loader)


if __name__ == '__main__':
  args = parse()
  main(args)

