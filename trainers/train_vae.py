import os
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from utils.misc import overrides
from .utils import *


class Trainer(BaseTrainer):
  def __init__(self, args, nets, opt, dataloader, vis, logger, device):
    super(Trainer, self).__init__(args, nets, opt, \
      dataloader, vis, logger, device)
    self.cur_cap = self.args.cap_min
  
  def get_cap_loss(self, kl, step):
    cap = (self.args.cap_max - self.args.cap_min)* \
      step/self.args.cap_iters
    self.cur_cap = min(cap, self.args.cap_max)
    return self.args.cap_coeff*torch.abs(self.cur_cap - kl)

  @overrides(BaseTrainer)
  def put_in_work(self):
    """Puts in work like a man possessed."""
    epochs = int(np.ceil(self.args.steps / len(self.dataloader)))
    step = 0
    
    if self.nets['vae'].in_dim[0] == 3: 
      recon_func = sse_loss
    else: recon_func = bce_loss

    for _ in range(epochs):
      for x in self.dataloader:
        step += 1
        if step > self.args.steps: return
    
        # 1. optimize VAE
        x = x.to(self.device)
        recon_batch, z, dist_params = self.nets['vae'](x)
        recon_loss = recon_func(recon_batch, x)
    
        # KL for gaussian
        kl_cont_dw = torch.empty(0).to(self.device)
        cont_cap_loss = 0
        if 'cont' in dist_params.keys():
          mu, logvar = dist_params['cont']
          kl_cont_dw = kl_gauss_unag(mu, logvar).mean(0)
          cont_cap_loss = self.get_cap_loss(kl_cont_dw.sum(), step)
                      
        # KL for categorical
        kl_cats = torch.empty(0).to(self.device)
        cat_cap_loss = 0
        if 'cat' in dist_params.keys():
          for logits in dist_params['cat']:
            kl_cat = kl_cat_unag(logits).sum(1).mean()
            kl_cats = torch.cat((kl_cats, kl_cat.view(1)))
          cat_cap_loss = self.get_cap_loss(kl_cats.sum(), step)
    
        vae_loss = recon_loss + cont_cap_loss + cat_cap_loss

        self.opt['vae'].zero_grad()
        vae_loss.backward(retain_graph=True)
        self.opt['vae'].step()
        
        # log...
        self.logger.log_val('recon_loss', recon_loss.item())
        self.logger.log_val('vae_loss', vae_loss.item())
        self.logger.log_val('cur_cap', self.cur_cap, runavg=False)
        self.logger.log_val('cont_kl', kl_cont_dw.data.cpu().numpy())
        self.logger.log_val('cat_kl', kl_cats.data.cpu().numpy())
  
        if not step % self.args.log_interval:
          self.logger.print(step)
        
        if not step % self.args.save_interval:
          save_ckpt(self.nets['vae'], self.args.logdir)
          self.logger.save(step)
        
          filename = 'traversal_' + str(step) + '.png'
          save_path = os.path.join(self.args.logdir, filename)
          self.vis.traverse(save_path)

          filename = 'recon_' + str(step) + '.png'
          save_path = os.path.join(self.args.logdir, filename)
          self.vis.recon(save_path)

