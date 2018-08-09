import os
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from utils.misc import overrides
from .utils import *


def permute_dims(z):
  B, _ = z.size()
  perm_z = []
  for z_j in z.split(1, 1):
    perm = torch.randperm(B).to(z.device)
    perm_z_j = z_j[perm]
    perm_z.append(perm_z_j)
  return torch.cat(perm_z, 1)


class Trainer(BaseTrainer):
  @overrides(BaseTrainer)
  def put_in_work(self):
    """Puts in work like a man possessed."""
    epochs = int(np.ceil(self.args.steps / len(self.dataloader)))
    step = 0

    bce = F.binary_cross_entropy_with_logits
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
        if 'cont' in dist_params.keys():
          mu, logvar = dist_params['cont']
          kl_cont_dw = kl_gauss_unag(mu, logvar).mean(0)
                      
        # KL for categorical
        kl_cats = torch.empty(0).to(self.device)
        if 'cat' in dist_params.keys():
          for logits in dist_params['cat']:
            kl_cat = kl_cat_unag(logits).sum(1).mean()
            kl_cats = torch.cat((kl_cats, kl_cat.view(1)))
      
        # total correlation term
        tc_loss = 0
        if 'tc_disc' in self.nets:
          joint_z_logits = self.nets['tc_disc'](z)
          ones = torch.ones(joint_z_logits.size()).to(self.device)
          tc_loss = -bce(joint_z_logits, ones) + bce(joint_z_logits, ones - 1)
              
        vae_loss = recon_loss + kl_cont_dw.sum() + \
          kl_cats.sum() + self.args.tc_coeff*tc_loss

        self.opt['vae'].zero_grad()
        vae_loss.backward(retain_graph=True)
        self.opt['vae'].step()
  
        # 4. optimize total correlation discriminator
        if 'tc_disc' in self.nets:
          z_prime = self.nets['vae'](x, decode=False)[1]
          z_pperm = permute_dims(z_prime).detach()
      
          tc_disc_loss = 0.5*(bce(joint_z_logits, ones) + \
            bce(self.nets['tc_disc'](z_pperm), ones - 1)) 
                          
          self.opt['tc_disc'].zero_grad()
          tc_disc_loss.backward()
          self.opt['tc_disc'].step()
        
        # log...
        self.logger.log_val('recon_loss', recon_loss.item())
        self.logger.log_val('vae_loss', vae_loss.item())
        self.logger.log_val('tc_loss', tc_loss.item())
        self.logger.log_val('cont kl', kl_cont_dw.data.cpu().numpy())
        self.logger.log_val('cat kl', kl_cats.data.cpu().numpy())
  
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

