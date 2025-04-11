import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import models.networks as networks
from .base_model import BaseModel
from models.modules.architecture import Get_ltpe
from models.modules.GramLoss import GramLoss4

logger = logging.getLogger('base')




class SPSRModel(BaseModel):
    def __init__(self, opt):
        super(SPSRModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            self.cri_pix = nn.L1Loss().to(self.device)

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            # if self.cri_fea:  # load VGG perceptual loss
            self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            # Branch_init_iters
            self.Branch_pretrain = train_opt['Branch_pretrain'] if train_opt['Branch_pretrain'] else 0
            self.Branch_init_iters = train_opt['Branch_init_iters'] if train_opt['Branch_init_iters'] else 1


            # gradient_pixel_loss
            if train_opt['gradient_pixel_weight'] > 0:
                self.cri_pix_grad = nn.MSELoss().to(self.device)
                self.l_pix_grad_w = train_opt['gradient_pixel_weight']
            else:
                self.cri_pix_grad = None


            # ltpe_pixel_loss
            if train_opt['ltpe_pixel_weight'] > 0:
                self.cri_pix_ltpe = nn.L1Loss().to(self.device)
                self.l_ltpe_pixel_w = train_opt['ltpe_pixel_weight']
            else:
                self.cri_pix_ltpe = None


            # G_grad pixel loss
            if train_opt['pixel_branch_weight'] > 0:
                l_pix_type = train_opt['pixel_branch_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix_branch = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix_branch = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_branch_w = train_opt['pixel_branch_weight']

            else:
                logger.info('Remove G_grad pixel loss.')
                self.cri_pix_branch = None

            # G_ltpe pixel loss
            if train_opt['ltpe_branch_weight'] > 0:
                l_pix_type = train_opt['ltpe_branch_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix_branch_ltpe = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix_branch_ltpe = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_branch_ltpe_w = train_opt['ltpe_branch_weight']

            else:
                logger.info('Remove G_ltpe pixel loss.')
                self.cri_pix_branch_ltpe = None

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            optim_params = []
            for k, v in self.netG.named_parameters():  # optimize part of the model

                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)


            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
            self.get_ltpe = Get_ltpe()
            self.gramloss4 = GramLoss4()

    def feed_data(self, data, need_HR=True):
        # LR,tensor
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)


    def optimize_parameters(self, step):
        # G

        if(self.Branch_pretrain): 
            if(step < self.Branch_init_iters):
                for k,v in self.netG.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=False
            else:
                for k,v in self.netG.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=True


        self.optimizer_G.zero_grad()

        self.fake_H_branch, self.fake_H, self.output, self.ltpe_SR = self.netG(self.var_L)
        self.var_H_ltpe = self.get_ltpe(self.var_H)
        self.fake_H_ltpe = self.get_ltpe(self.fake_H)
        self.var_ref_ltpe = self.get_ltpe(self.var_ref)

        

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # if step >= 70000:
            if self.cri_pix:  # pixel loss
                l_g_pix = 0.001 * self.cri_pix(self.output, self.var_H)
                l_g_total += l_g_pix

            if self.cri_pix_branch_ltpe:
                l_g_gram = self.gramloss4(self.fake_H, self.var_H)
                l_g_total += l_g_gram
                l_g_gram_ltpe_branch = self.gramloss4(self.fake_H_branch, self.var_H_ltpe)
                l_g_total += l_g_gram_ltpe_branch
                l_g_l1_ltpe_branch = 0.001 * self.cri_pix(self.fake_H_branch, self.var_H_ltpe)
                l_g_total += l_g_l1_ltpe_branch

            l_g_total.requires_grad_(True)
            l_g_total.backward()
            self.optimizer_G.step()
        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_pix_branch_ltpe:
                self.log_dict['l_g_gram'] = l_g_gram.item()
                self.log_dict['l_g_gram_ltpe_branch'] = l_g_gram_ltpe_branch.item()
                self.log_dict['l_g_l1_ltpe_branch'] = l_g_l1_ltpe_branch.item()



    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H_branch, self.fake_H, self.output, self.ltpe_SR = self.netG(self.var_L)

            
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['SR_branch'] = self.fake_H_branch.detach()[0].float().cpu()

        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.is_train:
            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)


    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
