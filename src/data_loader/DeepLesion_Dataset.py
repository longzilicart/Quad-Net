# DeepLesion_Dataset by longzilicart

from torch.utils.data import Dataset
import numpy as np
import torch
import sys
import os
sys.path.append("..")
from utils.ct_tools import CT_Preprocessing

class DeepLesion_MAR_Dataset(Dataset):
    '''DeepLesion_MAR_Dataset
    [!] you may need to rewrite to fit your data, the getitem rule part. 
    Args:
        - HU_norm: whether normalize to HU window.
    return:
        ma_sinogram, li_sinogram, gt_sinogram, ma_ct, li_ct, gt_ct, metal_trace, mask
    '''
    def __init__(self, root_dir, mode, HU_norm = True, ):
        self.root_dir = root_dir
        self.mode = mode
        self.HU_norm = HU_norm
        self.dataset_path = os.path.join(root_dir, mode)
        self.idx_list = self.get_indice_idx(os.path.join(self.dataset_path, 'ma_ct'))
        self.cttool = CT_Preprocessing()
        self.get_mask_trace()
        print('finish init DeepLesion_MAR_Dataset')

    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        [image_indice, metal_indice] = self.idx_list[idx]
        ma_sinogram = np.load(self.read_rule('ma_sinogram', image_indice, metal_indice),)
        li_sinogram = np.load(self.read_rule('li_sinogram', image_indice, metal_indice),)
        gt_sinogram = np.load(self.read_rule('gt_sinogram', image_indice, metal_indice),)
        ma_ct = np.load(self.read_rule('ma_ct', image_indice, metal_indice),) 
        li_ct = np.load(self.read_rule('li_ct', image_indice, metal_indice),)
        gt_ct = np.load(self.read_rule('gt_ct', image_indice,)) 
        # get dataset
        ma_sinogram = torch.from_numpy(ma_sinogram).unsqueeze(0)
        li_sinogram = torch.from_numpy(li_sinogram).unsqueeze(0)
        gt_sinogram = torch.from_numpy(gt_sinogram).unsqueeze(0)
        ma_ct = torch.from_numpy(ma_ct).unsqueeze(0)
        li_ct = torch.from_numpy(li_ct).unsqueeze(0)
        gt_ct = torch.from_numpy(gt_ct).unsqueeze(0)

        mask = torch.from_numpy(self.mask[:,:,int(metal_indice)]).unsqueeze(0)
        metal_trace = torch.from_numpy(self.metal_trace[:,:,int(metal_indice)]).unsqueeze(0)
        if self.HU_norm:
            ma_ct = self.simple_window(ma_ct)
            li_ct = self.simple_window(li_ct)
            gt_ct = self.simple_window(gt_ct)
        return ma_sinogram, li_sinogram, gt_sinogram, ma_ct, li_ct, gt_ct, metal_trace, mask

    def get_mask_trace(self,):
        mask_path = os.path.join(self.dataset_path, 'mask.npy')
        metal_trace_path = os.path.join(self.dataset_path, 'metal_trace.npy')
        self.mask = np.load(mask_path)
        self.metal_trace = np.load(metal_trace_path)

    @staticmethod
    def get_indice_idx(indice_folder):
        names = os.listdir(indice_folder)
        idx_list = []
        for name in names:
            name = name.split('.')[0].split('_')
            image_indice, metal_indice = name[-2], name[-1]
            idx_list.append([image_indice, metal_indice])
        return idx_list

    def read_rule(self, name, image_indice, metal_indice=None):
        '''- train_set/val_set
        - metal_trace.npy
        - mask.npy
        - ma_sinogram
            - ma_sinogram_{image_indice}_{metal_indice}.npy
        - li_sinogram
        - gt_sinogram_water
        - ma_ct
        - li_ct
        - gt_ct
            - gt_ct_{image_indice}.npy # 没有metal
        '''
        path = os.path.join(self.dataset_path, name)
        if metal_indice is not None:
            name = f'{name}_{image_indice}_{metal_indice}.npy'
        else: 
            name = f'{name}_{image_indice}.npy'
        return os.path.join(path, name)

    def simple_window(self, img):
        '''default CT window width=3000, center=500
        '''
        # return self.cttool.window_transform(self.cttool.miu2HU(img))
        return self.cttool.window_transform_torch(self.cttool.miu2HU(img))


class DeepLesion_MAR_Dataset_sino(DeepLesion_MAR_Dataset):
    '''FDMAR for sinogram network
    修改__getitem__不读取图片
    '''
    def __init__(self, root_dir, mode, HU_norm = True, ):
        super(DeepLesion_MAR_Dataset_sino, self).__init__(root_dir, mode, HU_norm)

    def __getitem__(self, idx):
        [image_indice, metal_indice] = self.idx_list[idx]
        # sinogram domain only
        ma_sinogram = np.load(self.read_rule('ma_sinogram', image_indice, metal_indice))
        li_sinogram = np.load(self.read_rule('li_sinogram', image_indice, metal_indice)) 
        gt_sinogram_water = np.load(self.read_rule('gt_sinogram_water', image_indice, metal_indice))
        # get dataset
        ma_sinogram = torch.from_numpy(ma_sinogram).unsqueeze(0)
        li_sinogram = torch.from_numpy(li_sinogram).unsqueeze(0)
        gt_sinogram_water = torch.from_numpy(gt_sinogram_water).unsqueeze(0)
        # mask = torch.from_numpy(self.mask[:,:,int(metal_indice)]).unsqueeze(0) 
        metal_trace = torch.from_numpy(self.metal_trace[:,:,int(metal_indice)]).unsqueeze(0)
        mask = torch.from_numpy(self.mask[:,:,int(metal_indice)]).unsqueeze(0)
        return ma_sinogram, li_sinogram, gt_sinogram_water, 0, 0, 0, metal_trace, mask

class DeepLesion_MAR_Dataset_image(DeepLesion_MAR_Dataset):
    '''FDMAR for image network
    修改__getitem__不读取图片
    '''
    def __init__(self, root_dir, mode, HU_norm = True, ):
        super(DeepLesion_MAR_Dataset_image, self).__init__(root_dir, mode, HU_norm)

    def __getitem__(self, idx):
        [image_indice, metal_indice] = self.idx_list[idx]
        # image domain only
        ma_ct = np.load(self.read_rule('ma_ct', image_indice, metal_indice),allow_pickle=True)
        li_ct = np.load(self.read_rule('li_ct', image_indice, metal_indice),allow_pickle=True)
        gt_ct = np.load(self.read_rule('gt_ct', image_indice,),allow_pickle=True)
        # get dataset
        ma_ct = torch.from_numpy(ma_ct).unsqueeze(0)
        li_ct = torch.from_numpy(li_ct).unsqueeze(0)
        gt_ct = torch.from_numpy(gt_ct).unsqueeze(0)
        # TODO 临时
        mask = torch.from_numpy(self.mask[:,:,int(metal_indice)]).unsqueeze(0)
        if self.HU_norm:
            ma_ct = self.simple_window(ma_ct)
            li_ct = self.simple_window(li_ct)
            gt_ct = self.simple_window(gt_ct)
        return 0, 0, 0, ma_ct, li_ct, gt_ct, 0, mask




if __name__ == '__main__':
    pass




