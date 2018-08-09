import argparse
import torch

import os; import sys; sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models import VAE
from utils.vis_utils import Visualizer

from loaders.celeba import CelebAUtil
from loaders.mnist import MNISTUtil
from loaders.dsprites import SpritesUtil
from loaders.blobs import BlobsUtil


DATASETS = {
  'celeba': [(3, 64, 64), CelebAUtil],
  'mnist': [(1, 32, 32), MNISTUtil],
  'dsprites': [(1, 64, 64), SpritesUtil],
  'blobs': [(1, 32, 32), BlobsUtil]}


def _get_dataset(data_arg):
  """Checks if the given dataset is available. If yes, returns
     the input dimensions and datautil."""
  if data_arg not in DATASETS:
    raise ValueError("Dataset not available!")
  img_size, datautil = DATASETS[data_arg]
  return img_size, datautil()


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vae-ckpt', type=str, required=True, dest='vae_ckpt_path')
  parser.add_argument('--save-path', type=str, default='traversal.png')
  parser.add_argument('--nb-trav', type=int, default=10,
                      help='number of samples to visualize on the \
                            traversal canvas.')
  parser.add_argument('--dataset', type=str, required=True,
                      help='The dataset to use for training \
                           (celeba | mnist | dsprites | blobs)')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  return parser.parse_args()


def main(args):
  torch.manual_seed(args.seed)

  # device placement
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  # load vae ckpt
  vae_ckpt = torch.load(args.vae_ckpt_path)
  vae = VAE(vae_ckpt['in_dim'], vae_ckpt['cont_dim'], 
    vae_ckpt['cat_dims'], vae_ckpt['temp']).to(device)
  vae.load_state_dict(vae_ckpt['state_dict'])
  vae.eval()

  _, datautil = _get_dataset(args.dataset)
  vis = Visualizer(vae, device, datautil.testdata, args.nb_trav)
  vis.traverse(args.save_path)


if __name__ == '__main__':
  args = parse()
  main(args)

