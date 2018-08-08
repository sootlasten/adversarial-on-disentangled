from torch import optim

from utils.base_runner import get_common_parser, base_runner 
from models import VAE, tc_disc
from trainers.train_factorvae import Trainer


@base_runner
def run(img_dims, cont_dim, cat_dims, args):
  nets = {
    'vae': VAE(img_dims, cont_dim, cat_dims, args.temp),
    'tc_disc': tc_disc(cont_dim + sum(cat_dims))
  }
  optimizers = {
    'vae': optim.Adam(nets['vae'].parameters(), lr=args.eta),
    'tc_disc': optim.Adam(nets['tc_disc'].parameters(), lr=args.eta)
  }
  return nets, optimizers


def get_args(parser):
  parser.add_argument('--tc-coeff', type=float, default=35,
                      help='coefficient for the total correlation term')
  return parser.parse_args()


if __name__ == '__main__':
  parser = get_common_parser()
  args = get_args(parser)
  run(args, Trainer)

