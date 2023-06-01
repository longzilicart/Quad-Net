import argparse
import sys
# sys.path.append("..")

import RIL.recon as RIL
from utils.ct_tools import *
# 2023新添加测试
from Model.QuadNetbase import *
from Model.QuadNet import *
from Model.Basic_FFC import *
# from .MAR_trainer import *
import tensorboard
import wandb

def get_parser():
    parser = argparse.ArgumentParser(description='Dudonet log')
    # 记录间隔参数 itlog_intv
    parser.add_argument('--itlog_intv', default=200, type=int,
                        help='log interval') 
    # log
    parser.add_argument('--not-save',default=False,action='store_true',
                        help = 'if yes, only output terminal')
    parser.add_argument('--log_root',type=str,required=True,
                       help = 'logger')
    parser.add_argument('--log_dir', type=str, required=True,
                       help = 'name of the log dir')
    parser.add_argument('--log_name',type=str,default='/log.txt',
                       help = 'name of the log file')
    # checkpoint
    parser.add_argument('--checkpoint_root', type=str, required=True,
                        help='')  
    # tensorboard
    parser.add_argument('--tensorboard_root', type=str, default='',
                       help = 'root dir of tensorboard')
    parser.add_argument('--tensorboard_dir', type=str, required=True,
                       help = 'dir name of tensorboard')
    # WANDB config 
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='MAR')
    parser.add_argument('--wandb_root', type=str, default='')
    # DDP
    parser.add_argument('--local_rank',default=-1,type=int,
                        help='rank for DDP')
    parser.add_argument('--dataset_root', required=True, type=str, help='root path of the dataset')
    # TODO 
    parser.add_argument('--dataset_name', default='size512', type=str,
                        help='dataset name for training')
    # dataloader
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch_size')    
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle of the dataloader')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num_workers of the dataloader')
    parser.add_argument('--drop_last', default=False, type=bool,
                        help='drop_last of the dataloader')
    #optimizer
    parser.add_argument('--lr', default=0.001, type=float,
                        help='学习率')    
    parser.add_argument('--beta1', default=0.5, type=float,
                        help='Adam的beta1参数')    
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Adam的beta2参数')    
    parser.add_argument('--epochs', default=200, type=int,
                        help='训练轮数')    
    #step_optimizer
    parser.add_argument('--step_size', default=30, type=int,
                        help='Adam的beta2参数')    
    parser.add_argument('--step_gamma', default=0.5, type=float,
                        help='训练轮数')    

    #args for resume training
    parser.add_argument('--resume', default=False, type=bool,
                        help = '是否使用之前的checkpoint')
    parser.add_argument('--resume_opt', default=False, type=bool,
                        help = '是否使用之前的checkpoint')
    parser.add_argument('--net_check_path', default='', type=str,
                        help='net_checkpoint的具体路径')
    parser.add_argument('--opt_check_path', default='', type=str,
                        help='optimizer_checkpoint的具体路径')
    parser.add_argument('--sino_check_path', default='', type=str,
                        help='sino_check_path的具体路径,split_refine')
    parser.add_argument('--image_check_path', default='', type=str,
                        help='image_check_path的具体路径,split_refine')   
    parser.add_argument('--refine_check_path', default='', type=str,
                        help='refine_check_path的具体路径,split_refine')

    # network args
    parser.add_argument('--mode', default='sino', type=str,
                        help='mode for trainer train [sino, image, refine, split_refine]')
    parser.add_argument('--sino_net', default='mask_Fourier', type=str,
                        help='select the sinogram net [mask_Fourier, Fourier, mp...]')
    parser.add_argument('--image_net', default='Fourier', type=str,
                        help='select the sinogram net, [Fourier, sunet]') 
    parser.add_argument('--refine_net', default='Fourier', type=str,
                        help='select the refine net.,Fourier,refine_unet,window_unet')
    parser.add_argument('--sino_ginout', default='0.75', type=float,
                    help='sino_ginout') 
    parser.add_argument('--image_ginout', default='0.5', type=float,
                    help='image_ginout') 
    parser.add_argument('--refine_ginout', default='0.5', type=float,
                    help='refine_ginout')
    parser.add_argument('--refine_f_block', default='s_unit', type=str,
                    help = 'refine_unet fourier block: select: s_unit, f_unit')
    parser.add_argument('--refine_c_block', default='vanilla', type=str,
                    help = 'refine_unet convolution block: select: identity,vanilla, bfn')

    parser.add_argument('--tester',default=False, type=bool,
                        help='whether use tester' )
    parser.add_argument('--tester_save_name',default='default_save', type=str,
                        help='tester_save名字' )
    parser.add_argument('--tester_save_image',default=False, type=bool,
                        help='whether samve tester name' )
    parser.add_argument('--tester_save_path',default='', type=str,
                        help='tester_save path' )
    parser.add_argument('--ablation',default='MAR', type=str,
                        help='具体用什么ablation设置。MAR正常MAR,MARres,MARli只替换trace部分,MARim:只有图片')
    return parser




def main(opt):
    '''选择网络,调用trainer训练'''
    sino_net = None
    image_net = None
    refine_net = None
    cttool = CT_Preprocessing()
    
    #sinogram net
    if opt.mode in ['sino', 'refine', 'split_refine']:
        if opt.sino_net == 'mask_Fourier': # FDMAR sino
            sino_net = Mask_Fourier_sino(FFCResNetGenerator, n_downsampling = 2, resnet_conv_ginout=opt.sino_ginout, n_blocks = 6, add_out_act = False)
        # [最新测试]
        elif 'AE' in opt.sino_net:
            init_down_ginout = 0 
            lama_ginout = opt.sino_ginout
            init_conv_kwargs = {'ratio_gin':init_down_ginout,'ratio_gout':init_down_ginout,'enable_lfu':False}
            downsample_conv_kwargs = {'ratio_gin':init_down_ginout,'ratio_gout':init_down_ginout,'enable_lfu':False}
            resnet_conv_kwargs = {'ratio_gin':lama_ginout,'ratio_gout':lama_ginout,'enable_lfu':False}
            if opt.sino_net == "Fourier_AENet_mp": # with metal projection
                sino_net = Fourier_Sino_AENet(2, 1, n_downsampling=2, e_blocks=5, d_blocks=1, init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=downsample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs, global_skip = False)
            else: 
                raise NotImplementedError('FAEnet select error')
        else:
            raise NotImplementedError('sinogram domain not implemented')

    # image net
    if opt.mode in ['image', 'refine', 'split_refine']:
        if opt.image_net == 'Image_Unet': # double unet
            image_net = Image_Unet(1, 1)
        elif opt.image_net == 'SImage_Unet':
            image_net = SImage_Unet(1, 1) 
        else:
            raise NotImplementedError('image domain not implemented')

    # refine net
    if opt.mode == 'refine' or opt.mode == 'split_refine':
        if opt.refine_net == 'Image_Fourier_Refine':
            # 其中F_block_select需要调整
            refine_net = Image_Fourier_Refine(2, 1, f_block = opt.refine_f_block, c_block = opt.refine_c_block)
        else:
            raise NotImplementedError('refine image domain not implemented')
    
    if opt.dataset_name == 'size512': # 【TODO 没提供选项】
        radon = RIL.radon
        image_radon = RIL.image_radon

    elif opt.ablation == 'MARres':
        MAR_net = MAR_net_residual(sino_net,image_net,refine_net,radon)
    elif opt.ablation == 'MARres_maskimageu_mp':
        MAR_net = MAR_net_residual_mp(sino_net,image_net,refine_net,radon) 

if __name__ == "__main__":

    pass


# currently busy, will update later ...
# trainer = MAR_Trainer(opt.mode, sino_net, image_net, refine_net, radon, image_radon, opt)
# print('finish training')
# if __name__ == "__main__":    
#     # os.environ["WANDB_MODE"] = "offline"
#     parser = get_parser()
#     opt = parser.parse_args()
#     main(opt)
