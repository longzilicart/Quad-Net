import numpy as np
# from sklearn.cluster import FeatureAgglomeration
import tqdm
import os
import itertools

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid as make_grid
# amp not support real-imag tensor, so not supported lama in torch1.7
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import wandb

from utils.ms_logger import *
from utils.ct_tools import *
from utils.sobel_loss import *
from utils.vgg_perceptual_loss import *
from RIL.recon import *
from data_loader.DeepLesion_Dataset import *
from Model.QuadNetbase import *
from Model.QuadNet import *
from utils.cal_acc import *

class MAR_Trainer_Basic:
    '''MAR Trainer Basic
    Model:
    log:
        logger/tensorboard/wandb
    Train:
        fit function(fit, train, val wait for implement)
    '''
    def __init__(self, mode = None, sino_net = None, image_net = None, refine_net = None, radon = None, image_radon = None, opt = None):
      
        assert opt is not None
        self.opt = opt
        if not opt.tester:
            # 'DAN' -> 'Contrast'
            assert mode in ['sino', 'image', 'refine', 'split_refine', 'Contrast']
        self.mode = mode
        self.sino_net = sino_net
        self.image_net = image_net
        self.refine_net = refine_net
        self.radon = radon
        self.image_radon = image_radon
        self.cttool = CT_Preprocessing()

        # logger
        self.checkpoint_path = os.path.join(opt.checkpoint_root,opt.log_dir)
        self.logger = loadLogger(opt)
        self.logger.info(f'{opt}')
        self.tb_writer = SummaryWriter(os.path.join(opt.tensorboard_root,opt.tensorboard_dir))
        if opt.local_rank == 0 and opt.use_wandb:
            print('wandb init')
            self.wandb_writer = self.wandb_init(opt)
        self.iter = 0
        self.itlog_intv = opt.itlog_intv
        self.num_epochs = opt.epochs
        self.epoch = 0
        self.rgb_dict = {'r':255,'g':0,'b':0} # red for mask

    @staticmethod
    def weights_init(m):
        # kaiming init
        classname = m.__class__.__name__                               
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)    

    def init_adam_optimizer(self, net):
        # initialize pytorch Adam optimizer // no return, self.opt, self.stepopt
        # self.optimizer = torch.optim.Adam(net.parameters(), lr = self.opt.lr, betas = (self.opt.beta1, self.opt.beta2))
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=0.001,)
        # self.optimizer = torch.optim.SGD(net.parameters(), lr=self.opt.lr, momentum=0.9)
        self.step_optimizer = StepLR(self.optimizer, step_size = self.opt.step_size, gamma=self.opt.step_gamma)
    
    # ---- save load checkpoint ----
    @staticmethod
    def save_checkpoint(param, path, name:str, epoch:int):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, name + '_{}_epoch.pkl'.format(epoch))
        torch.save(param, checkpoint_path)

    def save_model(self,save_opt=True):
        ''' save model by sino, image, refine
        '''
        epoch = self.epoch
        name = self.mode
        if self.mode in ['sino', 'image']:
            if self.mode == 'sino':
                net_check = self.sino_net.module.state_dict()
            elif self.mode == 'image':
                net_check = self.image_net.module.state_dict()
        # save FDMAR refine_net/split_refine_net seperately
        elif self.opt.contrast_mode == '' and (self.mode in ['refine', 'split_refine']):
            if self.opt.ablation in ['MARim','MARim_mask']:
                net_check = {
                    'sino':self.image_net.module.sino_net.state_dict(),
                    'image':self.image_net.module.image_net.state_dict(),
                    'refine':self.image_net.module.refine_net.state_dict()
                }
            else:
                net_check = {
                    'sino':self.MAR_net.module.sino_net.state_dict(),
                    'image':self.MAR_net.module.image_net.state_dict(),
                    'refine':self.MAR_net.module.refine_net.state_dict()
                }
        else:
            net_check = {
                'net':self.Dan_net.module.state_dict(),
            }
        self.save_checkpoint(net_check, self.checkpoint_path, name+'-net', epoch)

    def save_opt(self):
        '''basic save optimizer'''
        name = self.mode + 'opt'
        checkpoint_path = self.opt.checkpoint_root
        opt_param = self.optimizer.state_dict()
        step_opt_param = self.step_optimizer.state_dict() # if step opt is True
        opt_check = {
            'optimizer': opt_param,
            'step_optimizer': step_opt_param,
            'epoch' : self.epoch,
        }
        self.save_checkpoint(opt_check, checkpoint_path, name + '-opt', self.epoch)

    def load_model(self, strict = True):
        '''basic load model for **MAR**
        '''
        # strict = False
        print(f"load model strict is {strict}")
        if self.opt.contrast_mode == '': # 非对比实验
            if self.mode == 'sino':
                self.sino_net.load_state_dict(torch.load(self.opt.net_check_path,map_location='cpu'))
            elif self.mode == 'image':
                self.image_net.load_state_dict(torch.load(self.opt.net_check_path,map_location='cpu'))
            elif self.mode == 'refine':
                dict_checkpoint = torch.load(self.opt.net_check_path,map_location='cpu')
                self.MAR_net.sino_net.load_state_dict(dict_checkpoint['sino'], strict=strict)
                self.MAR_net.image_net.load_state_dict(dict_checkpoint['image'], strict=strict) 
                self.MAR_net.refine_net.load_state_dict(dict_checkpoint['refine'], strict=strict)
            elif self.mode == 'split_refine':
                if self.opt.sino_check_path != '':
                    self.MAR_net.sino_net.load_state_dict(torch.load(self.opt.sino_check_path, map_location='cpu'), strict=strict)
                if self.opt.image_check_path !='':
                    self.MAR_net.image_net.load_state_dict(torch.load(self.opt.image_check_path, map_location='cpu'), strict=strict)
                if self.opt.refine_check_path != '':
                    self.MAR_net.refine_net.load_state_dict(torch.load(self.opt.refine_check_path, map_location='cpu')['refine'], strict=strict)
                elif self.opt.refine_net not in ['unet', 'fourier_unet']:
                    try:
                        self.MAR_net.refine_net.apply(self.weights_init)
                        print('apply kaiming init for refine_net: {}'.format(self.opt.refine_net))
                    except Exception as err:
                        print(err)
        elif self.opt.contrast_mode in ['Dudo', 'Dudoplus']:
            if self.opt.net_check_path != '':
                self.Dan_net.load_state_dict(torch.load(self.opt.net_check_path,map_location='cpu')['net'])
            elif self.opt.mode == 'split_refine':
                if self.opt.refine_check_path != '':
                    self.Dan_net.image_net.load_state_dict(torch.load(self.opt.refine_check_path,map_location='cpu')['refine'])
                    print('dudo load refine checkpoint')
                else:
                    try:
                        self.Dan_net.image_net.apply(self.weights_init)  
                    except Exception as err:
                        print("loading dudo error", err)
                if self.opt.sino_check_path != '':
                    self.Dan_net.sino_net.load_state_dict(torch.load(self.opt.sino_check_path,map_location='cpu'))
                    print('dudo load sino checkpoint')
            print('finish loading')

        elif self.opt.contrast_mode == 'Dan':
            self.Dan_net.load_state_dict(torch.load(self.opt.net_check_path,map_location='cpu')['net'], strict = True)
            print('finish loading DAN')

    def load_opt(self):
        '''basic load opt'''
        opt_checkpath = self.opt.opt_checkpath
        opt_checkpoint = torch.load(opt_checkpath, map_location = 'cpu')
        self.optimizer.load_state_dict(opt_checkpoint['optimizer'])
        self.step_optimizer.load_state_dict(opt_checkpoint['step_optimizer'])
        self.epoch = opt_checkpoint['epoch']
        print('finish loading opt')

    def resume(self, mode = 'net', strict = True):
        '''resume training'''
        if self.opt.tester:
            strict = True
        if mode == 'net':
            if self.opt.resume:
                self.load_model(strict=strict)
                print('finish loading model')
        elif mode == 'opt':
            if self.opt.resume_opt and self.opt.opt_checkpath is not None:
                self.load_opt()
                print('finish loading opt')
        else:
            raise NotImplementedError('resume failed')

    # ---- common loss function ----
    @staticmethod
    def pixel_loss(input, target, mode = 'l1'):
        assert mode in ['l1', 'sml1', 'l2']
        if mode == 'l1':
            L1loss = torch.nn.L1Loss(reduction = 'mean')
            loss = L1loss(input, target)        
        elif mode == 'sml1':
            smL1loss = torch.nn.SmoothL1Loss(reduction = 'mean')
            loss = smL1loss(input, target)
        elif mode == 'l2':
            mse_loss = torch.nn.MSELoss(reduction = 'mean')
            loss = mse_loss(input, target)
        else:
            raise ValueError('pixel_loss error: mode not in [l1,sml1,l2]')
        return loss

    def sobel_loss(self, input, target):
        '''sobel loss
        intro:
            sobel filter to device
        usage:
            self.GradLoss = GradLoss()
            self.GradLoss.to(device) # gpu, else detach.tocpu -> calculate
        '''
        sobel_loss = self.GradLoss(input, target)
        return sobel_loss

    @staticmethod
    def mse_loss(input, target):
        return F.mse_loss(input, target)

    @staticmethod
    def perceptual_loss(mse_loss, blocks, weights, device):
        '''vgg perceptual loss
        mse_loss: func
        '''
        return FeatureLoss(mse_loss, blocks, weights, device)

    # modify the window
    @staticmethod
    def get_window_tensor(cttool, input_hu, width, center):
        ''' change window
        hu_img -> hu_img'''
        HUtensor = cttool.back_window_transform(input_hu, width=3000, center=500)
        wintensor = cttool.window_transform_torch(HUtensor, width=width, center=center)
        return wintensor

    @staticmethod    
    def get_window_tensor_from_miu(cttool, input_tensor, width=3000, center=500):
        '''miu_img -> hu_img'''
        window_tensor = cttool.window_transform_torch(cttool.miu2HU(input_tensor),width=width,center=center)
        return window_tensor
    
    def multi_window_transform(self, pred_hu, gt_hu, window_dict = None, ):
        ''' multi-window tranform for loss calculation
        input: pred_hu, gt_hu
        '''
        if window_dict is None:
            windows = [(3000, 500), (500, 50), (2000, 0)] # width, center
        else:
            windows = window_dict
        pred_hu = pred_hu.repeat((1, 3, 1, 1))
        gt_hu = gt_hu.repeat((1,3,1,1))
        pred_hu[:,1,:,:] = self.get_window_tensor(self.cttool, pred_hu[:,1,:,:], windows[1][0], windows[1][1])
        gt_hu[:,1,:,:] = self.get_window_tensor(self.cttool, gt_hu[:,1,:,:], windows[1][0], windows[1][1])
        pred_hu[:,2,:,:] = self.get_window_tensor(self.cttool, pred_hu[:,2,:,:], windows[2][0], windows[2][1])
        gt_hu[:,2,:,:] = self.get_window_tensor(self.cttool, gt_hu[:,2,:,:], windows[2][0], windows[2][1])
        return pred_hu, gt_hu

    def multi_window_loss(self, pred_hu, gt_hu, mask, window_dict = None, pixel_loss_mode = 'sml1'):
        pred_win = self.multi_window_transform(pred_hu, gt_hu, window_dict)
        gt_win = self.multi_window_transform(pred_hu, gt_hu, window_dict)
        loss = self.pixel_loss(pred_win * (1-mask), gt_win * (1-mask), mode = pixel_loss_mode)
        return loss

    def Quad_multi_window_loss(self, sino_out, image_out, radon_out, refine_out,
                                sino_gt, gt_ct, mask, metal_trace, percep_calculator, pixel_loss_mode = 'sml1', is_sobel = True, is_percep = True, is_freq=True): # window_mode = 'multi'
        ''' multi-window loss
        containing
            pixel-wise loss for [sino, radon, local_image,] between gt
            multi-win loss for [refine_image]
            multi-win sobel loss
            multi-win poerceptual loss based on VGG
        '''
        normal_window_dict = [(800, -600), (500, 50), (3000, 500)]
        perceptual_window_dict = [(3000, 500), (500, 50), (2000, 0)]
        refine_out_win, gt_ct_win = self.multi_window_transform(refine_out, gt_ct, normal_window_dict, )
        refine_out_winper, gt_ct_winper = self.multi_window_transform(refine_out, gt_ct, perceptual_window_dict, )
        
        # start calculating loss
        # 1. pixel loss for sino, radon, local_image
        sino_l = self.pixel_loss(sino_out * metal_trace, sino_gt * metal_trace, mode = pixel_loss_mode)
        # image_l = self.pixel_loss(image_out * (1 - mask), gt_ct * (1 - mask), mode = pixel_loss_mode)
        image_l = torch.tensor(0.0)
        radon_l = self.pixel_loss(radon_out * (1 - mask), gt_ct * (1 - mask), mode = pixel_loss_mode)
        # 2. window loss for refine out
        refine_win_pixel_l = self.pixel_loss(refine_out_win * (1 - mask), gt_ct_win * (1 - mask), mode = pixel_loss_mode)
        # 3. multi win sobel loss
        if is_sobel:
            refine_win_sobel_l = self.sobel_loss(refine_out_win * (1 - mask), gt_ct_win * (1 - mask))
        else:
            refine_win_sobel_l = torch.tensor(0.0)
        # 4. perceptural loss option
        if is_percep:
            refine_win_perceptual_l = percep_calculator(refine_out_winper * (1 - mask), gt_ct_winper * (1 - mask))
        else:
            # print("no percep")
            refine_win_perceptual_l = torch.tensor(0.0)
        if is_freq:
            focal_freq_l = self.focal_freq_loss(refine_out_win * (1 - mask), gt_ct_win * (1 - mask))
        else:
            focal_freq_l = torch.tensor(0.0)

        # cal the final loss
        refine_win_sobel_l = refine_win_sobel_l * 0.01
        refine_win_perceptual_l = refine_win_perceptual_l * 0.005
        refine_multi_win_l = refine_win_pixel_l + refine_win_sobel_l + refine_win_perceptual_l
        loss = sino_l + image_l + radon_l + refine_multi_win_l + focal_freq_l
        return loss, (sino_l, radon_l, image_l, refine_win_pixel_l,  refine_win_perceptual_l, refine_win_sobel_l, focal_freq_l)

    # ---- basic logging function ----
    @staticmethod
    def tensorboard_scalar(writer, r_path, step, **kwargs):
        ''' log kwargs by path automaticlly
        args:
            writer: self.writer
            r_path: /train/epoch/ + key
        kwargs:
            dict: {key: data}
        '''
        for key in kwargs.keys():
            path = os.path.join(r_path, key)
            # writer
            writer.add_scalar(path, kwargs[key], global_step=step)

    @staticmethod
    def tensorboard_image(writer, r_path, **kwargs):
        ''' log kwargs image by path automaticlly
        args: ...
        kwargs: 
            dict: {key: data}
        writer.add_image():
            (tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
        '''
        for key in kwargs.keys():
            path = os.path.join(r_path, key)
            writer.add_image(tag = path,
                            img_tensor = kwargs[key],
                            global_step = 0,
                            dataformats='CHW',)
    
    @staticmethod
    def wandb_init(opt):
        if opt.wandb_root == '':
            wandb_root = opt.tensorboard_root
        else:
            wandb_root = opt.wandb_root
        # keep the same rel path name with tensorboard
        wandb_dir = opt.tensorboard_dir
        wandb_path = os.path.join(wandb_root, wandb_dir)
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        wandb_writer = wandb.init(
                    project=opt.wandb_project,
                    name=str(wandb_dir),
                    config = opt,
                    dir=wandb_path,
                    resume = 'allow',
                    reinit = True,)
        return wandb_writer

    @staticmethod
    def to_wandb_img(**kwargs):
        # turn torch makegrid to wandb image
        for key, value in kwargs.items():
            kwargs[key] = wandb.Image(kwargs[key])
        return kwargs
    
    @staticmethod
    def wandb_logger(r_path, wandb_writer = None, step_name = None, step = None, **kwargs):
        '''
        r_path: the same as tensorboard, will be add before key
        step_name: [iter, epoch]
        step: int
        **kwargs: log info dict
        '''
        log_info = {}
        for key, value in kwargs.items():
            key_name = str(os.path.join(r_path, key))
            log_info[key_name] = kwargs[key]
        if step is not None:
            assert step is not None
            log_info.update({str(step_name): step})
        if wandb_writer is not None:
            wandb_writer.log(log_info)
        else:
            wandb.log(log_info)

    # reduce function
    def reduce_value(self, value, average=True):
        world_size = torch.distributed.get_world_size()
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.opt.local_rank)
        with torch.no_grad():
            dist.all_reduce(value)   # get reduce value
            if average:
                value /= world_size
        return value.cpu()

    # ---- training fucntion ----
    def fit():
        raise ValueError('function fit() not implemented')

    def train():
        pass

    def val():
        pass







# ---- MAR trainer ----

class MAR_Trainer(MAR_Trainer_Basic):
    '''
    '''
    def __init__(self, mode = None, sino_net = None, image_net = None, refine_net = None, radon = None, image_radon = None, opt = None):
        super(MAR_Trainer, self).__init__(mode, sino_net, image_net, refine_net, radon, image_radon, opt,)
        if mode in ['sino']:
            print('loading sinogram dataset')
            if not self.opt.tester:
                self.train_dataset = DeepLesion_MAR_Dataset_sino(opt.dataset_root,'train', HU_norm=False)
                self.val_dataset = DeepLesion_MAR_Dataset_sino(opt.dataset_root,'val', HU_norm=False)
            else:
                self.val_dataset = DeepLesion_MAR_Dataset_sino(opt.dataset_root,'val', HU_norm=True)
        elif mode in ['image']:
            print('loading image dataset')
            if not self.opt.tester:
                self.train_dataset = DeepLesion_MAR_Dataset_image(opt.dataset_root, 'train',)
            self.val_dataset = DeepLesion_MAR_Dataset_image(opt.dataset_root, 'val',)

        elif mode in ['image', 'refine', 'split_refine', 'Contrast', 'ma', 'NMAR']:
            # print('loading 512 * 512 image')
            if opt.mode in ['NMAR'] or opt.contrast_mode in ['NAMR', 'Dan', 'prior']:
                print('load not norm dataset')
                if not self.opt.tester:
                    self.train_dataset = DeepLesion_MAR_Dataset(opt.dataset_root,'train', HU_norm=False)
                self.val_dataset = DeepLesion_MAR_Dataset(opt.dataset_root,'val', HU_norm=False)
            else:
                print('load norm dataset')
                if not self.opt.tester:
                    self.train_dataset = DeepLesion_MAR_Dataset(opt.dataset_root,'train')
                self.val_dataset = DeepLesion_MAR_Dataset(opt.dataset_root,'val')
        else:
            raise ValueError('opt.dataset_name error : {}'.format(opt.dataset_name))
        
        # 【2】 define radon and image_radon
        self.radon = radon
        self.image_radon = image_radon


    #+++++++++++++++++++ fit main func +++++++++++++++++++ 
    def fit(self):
        if self.mode == 'sino':
            # self.fit_sino()
            pass
        elif self.mode == 'image':
            # self.fit_image()
            pass
        if self.mode in ['refine', 'split_refine']:
            if self.opt.ablation == 'MARres':
                self.MAR_net = MAR_net_residual(self.sino_net, self.image_net, self.refine_net, radon)
            elif self.opt.ablation == 'MARres_maskimageu_mp':
                self.MAR_net = MAR_net_residual_mp(self.sino_net, self.image_net, self.refine_net, radon)
            else:
                raise NotImplementedError
        self.fit_quadnet()

    def fit_quadnet(self):
        # fit quadnet
        opt = self.opt
        self.MAR_net = nn.SyncBatchNorm.convert_sync_batchnorm(self.MAR_net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', opt.local_rank)
        self.GradLoss = GradLoss()
        self.GradLoss.to(device)
        # resume training
        if opt.resume is True:
            # self.resume(mode='net')
            self.resume(mode='net', strict=False)
            print('finish loading checkpoint')
        self.MAR_net = self.MAR_net.to(device)
        self.MAR_net = torch.nn.parallel.DistributedDataParallel(self.MAR_net,
                                                device_ids = [opt.local_rank],
                                                output_device = opt.local_rank,
                                                find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  sampler = train_sampler,
                                  pin_memory = True,)
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1,
                                  num_workers=opt.num_workers,
                                  sampler = val_sampler,)
        self.init_adam_optimizer(self.MAR_net)
        if opt.resume_opt is True:
            self.resume(mode='opt')
            print('resume{}'.format(self.epoch))
        start_epoch = self.epoch
        self.iter = 0
        for self.epoch in range(start_epoch, opt.epochs):
            print(f"fit epoch: {self.epoch}")
            self.train_loader.sampler.set_epoch(self.epoch)
            self.train_quad()
            self.val_quad()
            self.step_optimizer.step()
            if opt.local_rank == 0:
                self.save_model(save_opt = True)

    #+++++++++++++++++++ train/val dual domain +++++++++++++++++++
    def train_quad(self,):
        datarange = 1
        losses, rmses, ssims, psnrs = [], [], [], []
        if self.opt.loss in ['multi', 'sobel']:
            percep_calculator = self.perceptual_loss(self.mse_loss, [0,1,2], [1,1,1], self.MAR_net.device)
        self.MAR_net.train()
        pbar = tqdm.tqdm(self.train_loader,ncols=60)
        for i,data in enumerate(pbar):
            ma_sinogram, li_sinogram, gt_sinogram_water, ma_ct, li_ct, gt_ct, metal_trace, mask = data
            ma_sinogram = ma_sinogram.to('cuda')
            metal_trace = metal_trace.to('cuda')
            gt_sinogram_water = gt_sinogram_water.to('cuda')
            ma_ct = ma_ct.to('cuda')
            gt_ct = gt_ct.to('cuda')
            mask = mask.to('cuda')
            if self.opt.ablation == 'MARres_maskimageu_mp':
                metal_proj = image_radon(mask)
                refine_out,(sino_out,image_out,sino_image) = self.MAR_net(ma_sinogram, metal_trace, ma_ct, mask, metal_proj)
            else: # MARres
                refine_out,(sino_out,image_out,sino_image) = self.MAR_net(ma_sinogram, metal_trace, ma_ct,)
            if self.opt.loss in ['multi']:
                loss, (sino_l, radon_l, image_l, refine_win_pixel_l, refine_win_perceptual_l,refine_win_sobel_l,focal_freq_l) = self.Quad_multi_window_loss(sino_out,image_out,sino_image,refine_out,gt_sinogram_water,gt_ct,mask, metal_trace, percep_calculator, pixel_loss_mode = 'sml1', is_sobel = True, is_percep = True, is_freq=False)
            else:
                raise NotImplementedError('loss')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            rmse,psnr,ssim = compute_measure(refine_out * (1-mask), gt_ct * (1-mask), datarange)
            rmses.append(self.reduce_value(rmse))
            ssims.append(self.reduce_value(ssim))
            psnrs.append(self.reduce_value(psnr))
            if self.opt.local_rank == 0:
                self.iter += 1
            if self.opt.local_rank == 0 and self.iter !=0 and self.iter % self.itlog_intv == 0:
                log_info = {
                    'loss': np.mean(losses[-self.itlog_intv:]),
                    'rmse': np.mean(rmses[-self.itlog_intv:]),
                    'ssim': np.mean(ssims[-self.itlog_intv:]),
                    'psnr': np.mean(psnrs[-self.itlog_intv:]),
                }
                if self.opt.loss in ['multi', 'single', 'sobel']:
                    log_info_2 = {'sino loss': sino_l,
                                'radon loss': radon_l,
                                'image loss': image_l,
                                'refine loss': refine_win_pixel_l,
                                'perceptual loss': refine_win_perceptual_l,
                                'sobel loss': refine_win_sobel_l,
                                "focal_freq_l":focal_freq_l,
                                }
                    log_info.update(log_info_2)
                img_info = {
                        'ma': make_grid(ma_ct,normalize=True),
                        'metal': make_grid(mask,normalize=True),
                        'image net out': make_grid(image_out*(1-mask),normalize=True),
                        'sino out': make_grid(sino_out,normalize=True),
                        'radon out': make_grid(sino_image*(1-mask),normalize=True),
                        'refine out': make_grid(refine_out*(1-mask),normalize=True),
                        'ground truth': make_grid(gt_ct*(1-mask),normalize=True),
                }
                self.tensorboard_scalar(self.tb_writer, 'train/loss', self.iter, **log_info)
                self.tensorboard_image(self.tb_writer, 'train', **img_info)
                if self.opt.use_wandb:
                    self.wandb_logger('train/iter', **log_info)

        if self.opt.local_rank == 0:
            # epoch info
            epoch_log = {
                'loss': np.mean(losses),
                'rmse': np.mean(rmses),
                'ssim': np.mean(ssims),
                'psnr': np.mean(psnrs),
            }
            img_info = {
                'ma': make_grid(ma_ct,normalize=True),
                'metal': make_grid(mask,normalize=True),
                'image net out': make_grid(image_out*(1-mask),normalize=True),
                'sino out': make_grid(sino_out,normalize=True),
                'radon out': make_grid(sino_image*(1-mask),normalize=True),
                'refine out': make_grid(refine_out*(1-mask),normalize=True),
                'ground truth': make_grid(gt_ct*(1-mask),normalize=True),
                }
            self.tensorboard_scalar(self.tb_writer, 'train/epoch', self.epoch, **epoch_log)
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.tensorboard_scalar(self.tb_writer, 'opt', self.epoch, **{'current_lr':current_lr,'batch_size':self.opt.batch_size})
            if self.opt.use_wandb:
                # wandb log
                epoch_log.update(self.to_wandb_img(**img_info))
                self.wandb_logger('train/epoch', wandb_writer = self.wandb_writer, step_name = 'epoch', step = self.epoch, **epoch_log)
                self.wandb_logger('settings', wandb_writer = self.wandb_writer, step_name = 'epoch', step = self.epoch, **{'current_lr':current_lr,'batch_size':self.opt.batch_size})

    def val_quad(self, val_mode='epoch'):
        print('start val_dudo for FDMAR')
        datarange = 1
        losses = []
        rmses = []
        ssims = []
        psnrs = []
        # percep_calculator
        if self.opt.loss in ['multi','single','sobel']:
            percep_calculator = self.perceptual_loss(self.mse_loss, [0,1,2], [1,1,1], self.MAR_net.device)
        # validation
        self.MAR_net.eval()
        pbar = tqdm.tqdm(self.val_loader,ncols=60)
        for i,data in enumerate(pbar):
            with torch.no_grad():
                ma_sinogram, li_sinogram, gt_sinogram_water, ma_ct, li_ct, gt_ct, metal_trace, mask = data
                ma_sinogram = ma_sinogram.to('cuda')
                metal_trace = metal_trace.to('cuda')
                gt_sinogram_water = gt_sinogram_water.to('cuda')
                ma_ct = ma_ct.to('cuda')
                gt_ct = gt_ct.to('cuda')
                mask = mask.to('cuda')
                gt_sinogram_water = gt_sinogram_water.to('cuda')
                if self.opt.ablation == 'MARres_maskimageu_mp':
                    metal_proj = image_radon(mask)
                    refine_out,(sino_out,image_out,sino_image) = self.MAR_net(ma_sinogram, metal_trace, ma_ct, mask, metal_proj)
                else: # MARres
                    refine_out,(sino_out,image_out,sino_image) = self.MAR_net(ma_sinogram, metal_trace, ma_ct,)
                if self.opt.loss in ['multi']:
                    loss, (sino_l, radon_l, image_l, refine_win_pixel_l, refine_win_perceptual_l,refine_win_sobel_l,focal_freq_l) = self.Quad_multi_window_loss(sino_out,image_out,sino_image,refine_out,gt_sinogram_water,gt_ct,mask, metal_trace, percep_calculator, pixel_loss_mode = 'sml1', is_sobel = True, is_percep = True, is_freq=False)
                else:
                    raise NotImplementedError('loss')
                
                # running acc
                losses.append(loss.item())
                rmse,psnr,ssim = compute_measure(refine_out * (1-mask), gt_ct * (1-mask), datarange)
                rmses.append(self.reduce_value(rmse))
                ssims.append(self.reduce_value(ssim))
                psnrs.append(self.reduce_value(psnr))

        # validation epoch acc
        if self.opt.local_rank == 0:
            epoch_log = {
                'loss': np.mean(losses),
                'rmse': np.mean(rmses),
                'ssim': np.mean(ssims),
                'psnr': np.mean(psnrs),
            }
            if self.opt.loss in ['multi', 'single', 'sobel']:
                log_info_2 = {'sino loss': sino_l,
                            'image loss': image_l,
                            'refine loss': refine_win_pixel_l,
                            'perceptual loss': refine_win_perceptual_l,
                            'sobel loss': refine_win_sobel_l,
                            'focal_freq_l': focal_freq_l}
                epoch_log.update(log_info_2)    
            img_info = {
                'ma': make_grid(ma_ct,normalize=True),
                'metal': make_grid(mask,normalize=True),
                'image net out': make_grid(image_out*(1-mask),normalize=True),
                'sino out': make_grid(sino_out,normalize=True),
                'radon out': make_grid(sino_image*(1-mask),normalize=True),
                'refine out': make_grid(refine_out*(1-mask),normalize=True),
                'ground truth': make_grid(gt_ct*(1-mask),normalize=True),
            }
            self.tensorboard_scalar(self.tb_writer, 'val/epoch', self.epoch, **epoch_log)
            self.tensorboard_image(self.tb_writer, 'val', **img_info)
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.tensorboard_scalar(self.tb_writer, 'opt', self.epoch, **{'current_lr':current_lr,'batch_size':self.opt.batch_size})
            if self.opt.use_wandb:
                epoch_log.update(self.to_wandb_img(**img_info))
                self.wandb_logger('val/epoch', wandb_writer = self.wandb_writer, step_name = 'epoch', step = self.epoch, **epoch_log)
                self.wandb_logger('settings', wandb_writer = self.wandb_writer, step_name = 'epoch', step = self.epoch, **{'current_lr':current_lr,'batch_size':self.opt.batch_size})
    



if __name__ == '__main__':
    pass

















