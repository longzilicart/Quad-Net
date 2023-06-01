import sys
import numpy as np
import torch
from torch import nn
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cg
import torch_radon



# 【512-640】 geometry
class Radon_Param_Basic:
    angle = 640         # sinogram angle num
    source_distance = 1075
    d_count = 640       # sinogram detectors num
    spacing = 0.08  
    img_size = 512      # image resolution
    start_angle_bias = np.pi * 2 / angle  
    # det_spacing = -1  # default 2.0
    det_spacing = 0.05
    imPixScale = 512 / 512 * 0.08

radon_param_basic = Radon_Param_Basic()

def radon(sinogram, sparse_angle = None, angle_bias = 0, img_size=None):
    ''' sinogram to CT_image
    intro: 
        project sinogram to CT image by ram-lak
    args:
        sinogram:
        sparse_angle: -> int
        angle_bias: -> float, calc by angle_list_w_bias()
    return:
        CT_image in miu = back_proj
    '''
    sinogram = sinogram / Radon_Param_Basic.imPixScale
    if sparse_angle is None:
        angle = radon_param_basic.angle
    else:
        angle = sparse_angle
    d_count = radon_param_basic.d_count
    angles = np.linspace(0, np.pi*2, angle, endpoint=False)
    
    # if angle_bias, angles with start angle bias
    if angle_bias != 0:
        # assert angle_bias < np.pi*2/angle
        angles = angle_list_w_bias(angles, angle_bias)
    
    if img_size is None:
        img_size = radon_param_basic.img_size

    source_distance = radon_param_basic.source_distance
    radon = RadonFanbeam(img_size,angles, source_distance, det_count = d_count, det_spacing = radon_param_basic.det_spacing)
    ma_rotate = sinogram
    filter_sin = radon.filter_sinogram(ma_rotate, "ram-lak")
    back_proj = radon.backprojection(filter_sin) 
    return back_proj 


def image_radon(image, sparse_angle = None, angle_bias = 0, img_size=None):
    ''' CT_image to sinogram
    intro: 
        construct CT image to sinogram
    args:
        image: CT image in valid miu
        sparse_angle: -> int
        angle_bias: -> float, calc by angle_list_w_bias()
    return:
        CT_image in miu = back_proj
    '''
    image = image * Radon_Param_Basic.imPixScale
    if sparse_angle is None:
        angle = radon_param_basic.angle
    else:
        angle = sparse_angle
    d_count = radon_param_basic.d_count
    angles = np.linspace(0, np.pi*2, angle, endpoint=False)
    
    # angles with start angle bias
    if angle_bias != 0:
        # assert angle_bias < np.pi*2/angle
        angles = angle_list_w_bias(angles, angle_bias)
    if img_size is None:
        img_size = radon_param_basic.img_size

    source_distance = radon_param_basic.source_distance
    radon = RadonFanbeam(img_size,angles,source_distance,det_count = d_count,det_spacing = radon_param_basic.det_spacing)
    sinogram = radon.forward(image)
    return sinogram

def angle_list_w_bias(angle_list, bias):
    def angle_w_bias(angle, bias):
        return angle + bias
    bias_list = len(angle_list) * [bias]
    angle_list = list(map(angle_w_bias, angle_list, bias_list))
    return angle_list

def add_poisson_to_sinogram_torch_fast(sinogram, IO, seed=None):
    max_sinogram = sinogram.max()
    sinogramRawScaled = sinogram / max_sinogram.max()
    sinogramCT = IO * torch.exp(-sinogramRawScaled)
    sinogram_CT_C = torch.zeros_like(sinogramCT)
    sinogram_CT_C = torch.poisson(sinogramCT)
    sinogram_CT_D = sinogram_CT_C / IO
    siongram_out = - max_sinogram * torch.log(sinogram_CT_D)
    return siongram_out


if __name__ == "__main__":
    pass
