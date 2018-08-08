from torch import optim

from utils.base_runner import get_common_parser, base_runner 
from models import VAE
from trainers.train_vae import Trainer


@base_runner
def run(img_dims, cont_dim, cat_dims, args):
  nets = {
    'vae': VAE(img_dims, cont_dim, cat_dims, args.temp)
  }
  optimizers = {
    'vae': optim.Adam(nets['vae'].parameters(), lr=args.eta, 
      weight_decay=args.weight_decay)
  }
  return nets, optimizers


def get_args(parser):
  parser.add_argument('--cap-coeff', type=float, default=30,
                      help='capacity constraint coefficient')
  parser.add_argument('--cap-min', type=float, default=0,
                      help='min capacity for KL')
  parser.add_argument('--cap-max', type=float, default=5,
                      help='max capacity for KL')
  parser.add_argument('--cap-iters', type=int, default=100000,
                      help='number of iters to increase the capacity over')
  return parser.parse_args()


if __name__ == '__main__':
  parser = get_common_parser()
  args = get_args(parser)
  run(args, Trainer)

