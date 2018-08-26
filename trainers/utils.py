import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn import functional as F

from utils.log_utils import Logger


def kl_gauss_unag(mu, logvar):
  kld = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
  return kld


def kl_cat_unag(logits):
  """The (unaggregated, i.e. not summed over dimensions yet) 
     KL divergence as per equation 22 in CONCRETE paper."""
  q_z = F.softmax(logits, dim=1)
  return q_z*(torch.log(q_z + 1e-20) - np.log(1/logits.size(1)))


def bce_loss(source, target):
  return F.binary_cross_entropy(source, target, reduce=False).mean(0).sum()


def sse_loss(source, target):
  return 0.5*(source - target).pow(2).mean(0).sum()


def save_ckpt(model, logdir):
  savepath = os.path.join(logdir, 'model.ckpt')
  torch.save({
    'in_dim': model.in_dim,
    'cont_dim': 0 if not len(model.cont_dim) else model.cont_dim[0],
    'cat_dims': model.cat_dims,
    'temp': model.temp,
    'state_dict': model.state_dict()
  }, savepath) 


class BaseTrainer(ABC):
  def __init__(self, args, nets, opt, dataloader, vis, logger, device):
    self.args = args
    self.nets = nets
    self.opt = opt
    self.dataloader = dataloader
    self.vis = vis
    self.logger = logger
    self.device = device
  
  @abstractmethod
  def put_in_work():
    pass

