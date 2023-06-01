# Model architecture of Quad-Net for metal artifact reduction

import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utils.ct_tools import CT_Preprocessing
from Model.QuadNetbase import *
from Model.Basic_unet import *



# =============== sinogram MAR ==================
class Mask_Fourier_sino(nn.Module):
    '''
    the same with Fourier_sino, use a binary mask to mask the metal trace
    '''
    def __init__(self, F_net, n_downsampling, resnet_conv_ginout, n_blocks=6, add_out_act = 'sigmoid'):
        super(Mask_Fourier_sino, self).__init__()
        init_conv_kwargs = {'ratio_gin':0,'ratio_gout':0,'enable_lfu':False}
        downsample_conv_kwargs = {'ratio_gin':0,
                                'ratio_gout':0,
                                'enable_lfu':False}
        resnet_conv_kwargs = {'ratio_gin':resnet_conv_ginout,
                                'ratio_gout':resnet_conv_ginout,
                                'enable_lfu':False}
        self.sino_net = F_net(input_nc = 2, output_nc = 1, n_blocks=n_blocks, n_downsampling=n_downsampling,
                                add_out_act=add_out_act,
                                init_conv_kwargs=init_conv_kwargs,
                                downsample_conv_kwargs = downsample_conv_kwargs,
                                resnet_conv_kwargs=resnet_conv_kwargs,
                                        )
                                #add_out_act='sigmoid'
    def forward(self, ma_sinogram, metal_trace):
        '''
            metal-corrupted region:1 otherwise:0
        '''
        h, w = ma_sinogram.size(-2), ma_sinogram.size(-1)
        if h != w:
            sino_in = torch.cat((ma_sinogram[:,:,:-1,:] * (1 - metal_trace[:,:,:-1,:]), metal_trace[:,:,:-1,:]), dim = 1)
            sino_cut = ma_sinogram[:,:,-1,:].unsqueeze(2)
        else:
            sino_in = torch.cat((ma_sinogram * (1 - metal_trace), metal_trace), dim = 1)
        sino_out = self.sino_net(sino_in)
        if h != w:
            sino_out = torch.cat((sino_out, sino_cut), dim=2)
        return sino_out


class Fourier_Sino_AENet(nn.Module):
    '''Fourier AENet'''    
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, 
                e_blocks=8, d_blocks = 1, 
                norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                add_out_act=False, max_features=1024, out_ffc=False, out_ffc_kwargs={}, global_skip = False):
        super(Fourier_Sino_AENet, self).__init__()
        self.encoder = FFC_Encoder(input_nc, output_nc, ngf=ngf,
            n_downsampling=n_downsampling, n_blocks = e_blocks, max_features=max_features,
            norm_layer=norm_layer,padding_type=padding_type, activation_layer=activation_layer,
            init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=downsample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs,
            add_out_act=add_out_act, )
        self.decoder = FFC_Decoder(input_nc, output_nc, ngf=ngf,
            n_downsampling=n_downsampling, n_blocks = d_blocks, norm_layer=norm_layer,
            padding_type=padding_type, activation_layer=activation_layer,
            up_norm_layer=up_norm_layer, up_activation=up_activation,
            resnet_conv_kwargs=resnet_conv_kwargs,
            add_out_act=add_out_act, max_features=max_features, out_ffc=out_ffc, out_ffc_kwargs=out_ffc_kwargs)
        self.global_skip = global_skip

    def forward(self, ma_sinogram, metal_trace):
        h, w = ma_sinogram.size(-2), ma_sinogram.size(-1)
        if h != w:
            sino_in = torch.cat((ma_sinogram[:,:,:-1,:], metal_trace[:,:,:-1,:]), dim=1)
            sino_cut = ma_sinogram[:,:,-1,:].unsqueeze(2)
        else:
            sino_in = torch.cat((ma_sinogram, metal_trace), dim=1) 
        latent_x = self.encoder(sino_in)
        sino_out = self.decoder(latent_x)
        if h != w:
            sino_out = torch.cat((sino_out, sino_cut), dim=2)
        if self.global_skip:
            sino_out = sino_out + ma_sinogram
        return sino_out


# ------------- MAR image net -----------------
# simple unet
class Image_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, ):
        super(Image_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output


class SImage_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, ):
        '''
        a simple image net
        '''
        super(SImage_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(self.n_channels + 1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        '''input x应该是ma_CT'''
        input = torch.cat((x, x), dim=1)
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        # output.clamp_(0, 1)
        # output = torch.clamp(output, min=0.0, max=1.0)
        return output


#  refinement
class Image_Fourier_Refine(nn.Module):
    def __init__(self,in_channels, out_channels, bilinear= True, 
                        f_block = 's_unit', c_block='vanilla'):
        super(Image_Fourier_Refine, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        self.upx1_fkc = Fourier_Skip_Block(in_channels=64,out_channels=64,
                        F_block_select=f_block,C_block_select=c_block,residual=True)
        self.upx2_fkc = Fourier_Skip_Block(in_channels=128,out_channels=128,
                        F_block_select=f_block,C_block_select=c_block,residual=True)
        self.upx3_fkc = Fourier_Skip_Block(in_channels=256,out_channels=256,
                        F_block_select=f_block,C_block_select=c_block,residual=True)
                        
    def forward(self, x):
        # step1： 正常downsample
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        bottle = self.down3(x3)
        x3 = self.upx3_fkc(x3)
        x2 = self.upx2_fkc(x2)
        x1 = self.upx1_fkc(x1)
        x = self.up1(bottle, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)
        output.clamp_(0, 1) # [!] depend on your need, if train with MIU, delete this
        return output




#++++++++++++++++++++++++++++ dual domain MAR ++++++++++++++++++++++++++++++
# In application, one can modify the window_transform part or just delete it. 
# it may yield better with appropriate optimizations.

class MAR_net(nn.Module):
    '''Quad domain FMAR without residual
    image_net ------- concat --- refine_net
    sino_net -> radon  -^
    '''
    def __init__(self, sino_net, image_net, refine_net, radon):
        super(MAR_net, self).__init__()
        self.sino_net = sino_net
        self.image_net = image_net
        self.refine_net = refine_net
        self.radon = radon
        self.cttool = CT_Preprocessing()
        
    def forward(self, sino_in, metal_trace, ma_image):
        sino_out = self.sino_net(sino_in, metal_trace)
        image_out  = self.image_net(ma_image)
        sino_image = self.radon(sino_out)
        sino_out = sino_out * metal_trace + sino_in * (1-metal_trace)
        sino_image = self.cttool.window_transform_torch(self.cttool.miu2HU(sino_image))
        refine_in = torch.cat((image_out, sino_image), dim=1)
        refine_out = self.refine_net(refine_in)
        return refine_out,(sino_out,image_out,sino_image)

class MAR_LI_net(nn.Module):
    '''Quad FMAR with LI as skip connection 
    [note] it may get over-smoothing results around the metal
    '''
    def __init__(self, sino_net, image_net, refine_net, radon):
        super(MAR_LI_net, self).__init__()
        self.sino_net = sino_net
        self.image_net = image_net
        self.refine_net = refine_net
        self.radon = radon
        self.cttool = CT_Preprocessing()
        
    def forward(self, sino_in, metal_trace, li_image,):
        sino_out = self.sino_net(sino_in, metal_trace)
        sino_out = sino_out * metal_trace + sino_in * (1-metal_trace)
        image_out  = li_image
        sino_image = self.radon(sino_out)
        sino_image = self.cttool.window_transform_torch(self.cttool.miu2HU(sino_image))
        refine_in = torch.cat((image_out, sino_image), dim=1)
        refine_out = self.refine_net(refine_in) # + image_out
        return refine_out, (sino_out, image_out, sino_image)

class MAR_net_residual(nn.Module):
    def __init__(self, sino_net, image_net, refine_net, radon):
        super(MAR_net_residual, self).__init__()
        self.sino_net = sino_net
        self.image_net = image_net
        self.refine_net = refine_net
        self.radon = radon 
        self.cttool = CT_Preprocessing()
        
    def forward(self, sino_in, metal_trace, ma_image, ):
        #sino_net
        sino_out = self.sino_net(sino_in, metal_trace)
        #image_net
        image_out  = self.image_net(ma_image)
        sino_out = sino_out * metal_trace + sino_in * (1-metal_trace)
        sino_image = self.radon(sino_out)
        sino_image = self.cttool.window_transform_torch(self.cttool.miu2HU(sino_image))
        #refine
        refine_in = torch.cat((image_out, sino_image), dim=1)
        # with residual
        refine_out = self.refine_net(refine_in) + image_out
        # no residual [select MAR]
        return refine_out,(sino_out, image_out, sino_image)

class MAR_net_residual_mp(nn.Module):
    '''MAR residual with mask image unet'''
    def __init__(self, sino_net, image_net, refine_net, radon):
        super(MAR_net_residual_mp, self).__init__()
        self.sino_net = sino_net
        self.image_net = image_net
        self.refine_net = refine_net
        self.radon = radon
        self.cttool = CT_Preprocessing()
        
    def forward(self, sino_in, metal_trace, ma_image, metal_proj):
        sino_out = self.sino_net(sino_in, metal_proj)
        image_out  = self.image_net(ma_image,)
        sino_out = sino_out * metal_trace + sino_in * (1-metal_trace)
        sino_image = self.radon(sino_out)
        sino_image = self.cttool.window_transform_torch(self.cttool.miu2HU(sino_image))
        #refine
        refine_in = torch.cat((image_out, sino_image), dim=1)
        refine_out = self.refine_net(refine_in) + image_out
        return refine_out, (sino_out, image_out, sino_image)


if __name__ == "__main__":
    sino_ginout = 0.75
    refine_f_block = 'f_unit'
    refine_c_block = 'identity'
    
    sino_net = Mask_Fourier_sino(FFCResNetGenerator, n_downsampling = 2, resnet_conv_ginout=sino_ginout, n_blocks = 6, add_out_act = False)
    img_net = SImage_Unet(1, 1) 
    refine_net = Image_Fourier_Refine(2, 1, f_block = refine_f_block, c_block=refine_c_block)


    # test model 
    sino_img, metal_trace = torch.randn(1,1,640,640), torch.randn(1,1,640,640)
    img, mask = torch.randn(1,1,512,512), torch.randn(1,1,512,512)

    print(sino_net(sino_img, metal_trace))
    print(img_net(img))
    print(refine_net(torch.cat((img, img), dim=1)))