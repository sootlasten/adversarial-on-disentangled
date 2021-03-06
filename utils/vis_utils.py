import os
import random
from functools import wraps

import numpy as np
import torch
from torchvision.utils import save_image


def temp_eval(func):
  @wraps(func)
  def wrapper(self, *args, **kwargs):
    self.model.eval()
    func(self, *args, **kwargs)
    self.model.train()
  return wrapper


class Visualizer():
  def __init__(self, model, device, dataset, nb_trav, indices=None):
    self.model = model
    self.device = device
    self.dataset = dataset
    self.trav_imgs = self._get_trav_imgs(nb_trav, indices)

  def _get_trav_imgs(self, nb_trav, indices):
    if not indices:
      indices = random.sample(range(1, len(self.dataset)), nb_trav)
    imgs = []
    for i, img_idx in enumerate(indices):
      img = self.dataset[img_idx]
      imgs.append(img)
    return torch.stack(imgs).to(self.device)

  @temp_eval
  def recon(self, save_path):
    imgs = []
    for i in range(50):
      img = self.dataset[i]
      imgs.append(img)
    imgs = torch.stack(imgs).to(self.device)
    recons, _, _ = self.model(imgs)
    canvas = torch.cat((imgs, recons), dim=0)
    save_image(canvas, save_path, nrow=10, pad_value=1)


  @temp_eval
  def traverse(self, save_path):
    recons, z, dist_params = self.model(self.trav_imgs)
    canvas = torch.cat((self.trav_imgs, recons), dim=0)
    nb_cols = len(self.trav_imgs)
      
    # traverse continuous 
    if 'cont' in dist_params:
      first_mu = dist_params['cont'][0][0]
      cont_dim = len(first_mu)
      latents = torch.cat((first_mu, z[0, cont_dim:]))
      
      trav_range = torch.linspace(-3, 3, nb_cols)
      for i in range(cont_dim):
        temp_latents = latents.repeat(nb_cols, 1)
        temp_latents[:, i] = trav_range
        recons = self.model.decoder(temp_latents)
        canvas = torch.cat((canvas, recons), dim=0)
    else:
      cont_dim = 0
      latents = z[0] 
    
    # traverse categorical
    if 'cat' in dist_params:
      for i, d in enumerate(self.model.cat_dims):
        si = cont_dim + sum(self.model.cat_dims[:i])
        d -= max(0, d - nb_cols)  # cut traversals off if not enough cols
        temp_latents = latents.repeat(d, 1)
        temp_latents[:, si: si + d] = torch.eye(d)
        recon = self.model.decoder(temp_latents)
        row = torch.ones(nb_cols, *recon.shape[1:]).to(self.device)
        row[:d] = recon
        canvas = torch.cat((canvas, row), dim=0)
    
    save_image(canvas, save_path, nrow=nb_cols, pad_value=1)

