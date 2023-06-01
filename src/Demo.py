from Main_trainer import *

radon = RIL.radon
image_radon = RIL.image_radon

sino_ginout = 0.75
refine_f_block = 'f_unit'
refine_c_block = 'vanilla'
# QuadNet
sino_net = Mask_Fourier_sino(FFCResNetGenerator, n_downsampling = 2, resnet_conv_ginout=sino_ginout, n_blocks = 6, add_out_act = False)
image_net = SImage_Unet(1, 1) 
refine_net = Image_Fourier_Refine(2, 1, f_block = refine_f_block, c_block=refine_c_block)
radon = RIL.radon
Quad_Net = MAR_net_residual(sino_net,image_net,refine_net,radon)
Quad_Net_mp = MAR_net_residual_mp(sino_net,image_net,refine_net,radon)


if __name__ == '__main__':
    sino_in, metal_trace = torch.randn(1,1,640,640), torch.randn(1,1,640,640)
    ma_image, mask = torch.randn(1,1,512,512), torch.randn(1,1,512,512)
    sino_in, metal_trace, ma_image, mask = sino_in.cuda(), metal_trace.cuda(), ma_image.cuda(), mask.cuda()
    Quad_Net = Quad_Net.cuda()
    Quad_Net_mp = Quad_Net_mp.cuda()

    # output
    with torch.no_grad():
        refine_out, (sino_out, image_out, sino_image) = Quad_Net(sino_in, metal_trace, ma_image)
        print(f"refine_out_shape{refine_out.shape}, sino_out_shape{sino_out.shape}, image_out_shape{image_out.shape}, sino_image_shape{sino_image.shape}")

        metal_proj = image_radon(mask)
        refine_out, (sino_out, image_out, sino_image) = Quad_Net_mp(sino_in, metal_trace, ma_image, metal_proj)
        print(f"mprefine_out_shape{refine_out.shape}, mpsino_out_shape{sino_out.shape}, mpimage_out_shape{image_out.shape}, mpsino_image_shape{sino_image.shape}")





