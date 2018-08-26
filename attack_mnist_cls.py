import time
import argparse
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal
from torchvision.utils import save_image

from models import VAE, MnistClassifier
from loaders.mnist_single_class import get_mnist_singlecls_loader
from loaders.mnist import MNISTUtil


def l2_norm(x):
  """Takes the L2 norm of the input tensor across all dimensions 
     except the 0-th, the latter of which is summer across."""
  return x.view(len(x), -1).pow(2).sum(1).sqrt().sum()


class Attacker():
  def __init__(self, model, args, device):
    self.args = args
    self.device = device
    self.model = model
    # gradients w.r.t. weights not needed
    for p in self.model.parameters(): p.requires_grad = False
  
  def vis_pair(self, src_class, tar_class):
    src_loader = get_mnist_singlecls_loader(src_class, 1)
    
    for src in src_loader:
      src = src.to(self.device)
      src_out = self.model(src).to(self.device)
      target = torch.tensor([tar_class]).to(self.device)
      adv = self._attack(src, target)
      adv_out = self.model(adv)

      print("src {:.2f}".format(src_out.exp() \
        .squeeze()[src_class]))
      print("tar {:.2f}".format(adv_out.exp() \
        .squeeze()[tar_class]))

      f, axarr = plt.subplots(2, 1)
      axarr[0].axis('off')
      axarr[1].axis('off')
      axarr[0].imshow(src.squeeze().data, cmap='gray')
      axarr[1].imshow(adv.squeeze().data, cmap='gray')
      plt.show()
        
  def _attack(self, src, tar):
    noise = Normal(loc=src, scale=self.args.noise_std)
    adv = torch.clamp(src + noise.sample(), 0, 1).requires_grad_()
    optimizer = optim.Adam([adv], lr=self.args.eta)
  
    # per-pixel lower and upper bounds
    _min = torch.max(torch.zeros_like(src.data), src.data-self.args.eps)
    _max = torch.min(torch.ones_like(src.data), src.data+self.args.eps)
    
    for i in range(self.args.steps):
      adv_out = self.model(adv)
      l2pix = l2_norm(adv - src)
      cls_loss = F.nll_loss(adv_out, tar)
      
      loss = self.args._lambda*l2pix + cls_loss
      loss.backward()
      optimizer.step()

      adv.data = torch.min(torch.max(adv.data, _min), _max)
        
    return adv
    

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', type=str, default=None, required=True)
  parser.add_argument('--vis-pair', type=str, default='', required=True,
                      help='a pair of digit classes to visualize the adversarial \
                            examples of')
  parser.add_argument('--eta', type=float, default=1e-3)
  parser.add_argument('--batch-size', type=int, default=64)
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
  parser.add_argument('--info-path', type=str, default='adversarial-info.txt')
  return parser.parse_args()


def run(args):
  torch.manual_seed(args.seed)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  model = MnistClassifier().to(device)
  state_dict = torch.load(args.ckpt)
  model.load_state_dict(state_dict)

  attacker = Attacker(model, args, device)
  try:
    src, tar = map(lambda x: int(x), args.vis_pair.split(','))
  except:
    raise ValueError("Argument 'vis-pair' needs to be a pair of digits")
  attacker.vis_pair(src, tar)


if __name__ == '__main__':
  args = parse()
  run(args)

