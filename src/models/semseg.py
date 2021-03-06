# -*- coding: utf-8 -*-

import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import infnet, fcn8_vgg16, unet_resnet, resnet_seam
from src import utils as ut
from src import models
from . import losses
import sys

try:
    import kornia
    from kornia.augmentation import RandomAffine
    from kornia.geometry.transform import flips
except:
    print('kornia not installed')
    
from scipy.ndimage.filters import gaussian_filter
from optimizers import adasls, sls, sps
from . import metrics, networks

class SemSeg(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict = exp_dict
        self.train_hashes = set()
        self.n_classes = self.exp_dict['model'].get('n_classes', 1)

        self.epoch = 0

        self.model_base = networks.get_network(self.exp_dict['model']['base'],
                                              n_classes=self.n_classes,
                                              exp_dict=self.exp_dict)
        self.cuda()
        opt_dict = self.exp_dict['optimizer']
        name = opt_dict['name']
        if name == "adam":
            opt = torch.optim.Adam(
                    self.model_base.parameters(), lr=opt_dict["lr"], betas=(0.99, 0.999))

        elif name == "sgd":
            opt = torch.optim.SGD(
                self.model_base.parameters(), lr=opt_dict["lr"], momentum=opt_dict['momentum'], weight_decay=opt_dict['weight_decay'])

        elif name == "sps":
            opt = sps.Sps(
                self.model_base.parameters(), c=opt_dict['c'])
        elif name == "adasls":
            opt = adasls.AdaSLS(
                self.model_base.parameters(), c=opt_dict['c'], momentum=opt_dict.get('momentum', 0.))

        elif name == "sls":
            opt = sls.Sls(
                self.model_base.parameters(), c=.5)
                               
        self.opt = opt


    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict(),
                      'epoch':self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        if 'opt' not in state_dict:
            return
        self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict['epoch']

    def train_on_loader(self, train_loader):
        
        self.train()
        self.epoch += 1
        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()
    
        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_monitor.add(score_dict)
            msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            pbar.update(1)
            
        pbar.close()

        return train_monitor.get_avg_score()

    def train_on_batch(self, batch, roi_mask=None):
        # add to seen images
        for m in batch['meta']:
            self.train_hashes.add(m['hash'])

        self.opt.zero_grad()

        images = batch["images"].cuda()
        
        # compute loss
        loss_name = self.exp_dict['model']['loss']
        if loss_name in 'cross_entropy':
            logits = self.model_base(images)
            # full supervision
            loss_func = lambda:losses.compute_cross_entropy(images, logits, masks=batch["masks"].cuda(), roi_masks=roi_mask)
        
        elif loss_name in 'point_level':
            logits = self.model_base(images)
            # point supervision
            loss_func = lambda:losses.compute_point_level(images, logits, point_list=batch['point_list'])
        
        elif loss_name in 'point_level':
            # implementation needed
            loss_func = lambda:losses.compute_const_point_loss(images, logits, point_list=batch['point_list'])

        # create loss closure
        def closure():
            loss = loss_func()
            if self.exp_dict['optimizer']['name'] not in ['adasls', 'sls']:
                loss.backward()
            return loss

        # update parameters
        self.opt.zero_grad()
        loss = self.opt.step(closure=closure)

            

        return {'train_loss': float(loss)}

    @torch.no_grad()
    def predict_on_batch(self, batch):
        self.eval()
        image = batch['images'].cuda()
    
        if self.n_classes == 1:
            res = self.model_base.forward(image)
            if 'shape' in batch['meta'][0]:
                res = F.upsample(res, size=batch['meta'][0]['shape'],              
                            mode='bilinear', align_corners=False)
            res = (res.sigmoid().data.cpu().numpy() > 0.5).astype('float')
        else:
            self.eval()
            logits = self.model_base.forward(image)
            res = logits.argmax(dim=1).data.cpu().numpy()

        return res 

    def vis_on_batch(self, batch, savedir_image, return_image=False):
        image = batch['images']
        original = hu.denormalize(image, mode='rgb')[0]
        gt = np.asarray(batch['masks'])

        image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        img_pred = hu.save_image(savedir_image,
                    original,
                      mask=self.predict_on_batch(batch), return_image=True)

        img_gt = hu.save_image(savedir_image,
                     original,
                      mask=gt, return_image=True)
        img_gt = models.text_on_image( 'Groundtruth', np.array(img_gt), color=(0,0,0))
        img_pred = models.text_on_image( 'Prediction', np.array(img_pred), color=(0,0,0))
        
        if 'points' in batch:
            pts = (batch['points'][0].numpy().copy() != 0).astype('uint8')
            # pts[pts == 1] = 2
            # pts[pts == 0] = 1
            # pts[pts == 255] = 0
            img_gt = np.array(hu.save_image(savedir_image, img_gt/255.,
                                points=pts.squeeze(), radius=2, return_image=True))
        img_list = [np.array(img_gt), np.array(img_pred)]
        if return_image:
            return img_list
        hu.save_image(savedir_image, np.hstack(img_list))
        # hu.save_image('.tmp/pred.png', np.hstack(img_list))

    def val_on_loader(self, loader, savedir_images=None, n_images=0):
        self.eval()
        val_meter = metrics.SegMeter(split=loader.dataset.split)
        i_count = 0
        for i, batch in enumerate(tqdm.tqdm(loader)):
            # make sure it wasn't trained on
            for m in batch['meta']:
                assert(m['hash'] not in self.train_hashes)

            val_meter.val_on_batch(self, batch)
            if i_count < n_images:
                self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images, 
                    '%d.png' % batch['meta'][0]['index']))
                i_count += 1

        
        return val_meter.get_avg_score()
        
    @torch.no_grad()
    def compute_uncertainty(self, images, replicate=False, scale_factor=None, n_mcmc=20, method='entropy'):
        self.eval()
        set_dropout_train(self)

        # put images to cuda
        images = images.cuda()
        _, _, H, W= images.shape

        if scale_factor is not None:
            images = F.interpolate(images, scale_factor=scale_factor)
        # variables
        input_shape = images.size()
        batch_size = input_shape[0]

        if replicate and False:
            # forward on n_mcmc batch      
            images_stacked = torch.stack([images] * n_mcmc)
            images_stacked = images_stacked.view(batch_size * n_mcmc, *input_shape[1:])
            logits = self.model_base(images_stacked)
            

        else:
            # for loop over n_mcmc
            logits = torch.stack([self.model_base(images) for _ in range(n_mcmc)])
            
            logits = logits.view(batch_size * n_mcmc, *logits.size()[2:])

        logits = logits.view([n_mcmc, batch_size, *logits.size()[1:]])
        _, _, n_classes, _, _ = logits.shape
        # binary do sigmoid 
        if n_classes == 1:
            probs = logits.sigmoid()
        else:
            probs = F.softmax(logits, dim=2)

        if scale_factor is not None:
            probs = F.interpolate(probs, size=(probs.shape[2], H, W))

        self.eval()


        if method == 'entropy':
            score_map = - xlogy(probs).mean(dim=0).sum(dim=1)

        if method == 'bald':
            left = - xlogy(probs.mean(dim=0)).sum(dim=1)
            right = - xlogy(probs).sum(dim=2).mean(0)
            bald = left - right
            score_map = bald


        return score_map 



class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}

def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))