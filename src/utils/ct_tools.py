import torch
import numpy as np
import torchvision


class CT_Preprocessing:
    '''
        1. miu to HU
        2. transform to window
    '''
    def __init__(self, miuwater=0.192):
        self.miuwater = miuwater

    def HU2miu(self, HUimg):
        miuimg = HUimg / 1000 * self.miuwater + self.miuwater
        return miuimg

    def miu2HU(self, miuimg):
        HUimg = (miuimg - self.miuwater) / self.miuwater * 1000
        return HUimg

    def CTrange(self, img, HUmode=True, minran=-1000, maxran=2000):
        assert minran < maxran
        if HUmode is False: #如果传入的是miu,先转成HU
            img = self.miu2HU(img)
        img[img<minran] = minran
        img[img>maxran] = maxran
        if HUmode is False:
            img = self.HU2miu(img)
        return img
    
    def window_transform(self, HUimg, width=3000, center=500, normal=False):
        '''
        transform to [0, 1]
        if normal, transform to [0, 255] -> may provide better optimization empirically.
        '''
        minwindow = float(center) - 0.5*float(width)
        winimg = (HUimg - minwindow)/float(width)
        winimg[winimg<0]=0
        winimg[winimg>1]=1
        if normal:
            print('normalize to 0-255')
            winimg = (winimg*255).astype('float')
        return winimg
    
    def window_transform_torch(self, HUimg, width=3000, center=500, normal=False):
        '''
        transform to [0, 1]
        if normal, transform to [0, 255] -> may provide better optimization empirically.
        '''
        minwindow = float(center)- 0.5 * float(width)
        winimg = (HUimg - minwindow) / float(width)
        winimg.clamp_(min = 0.0, max = 1.0)
        if normal:
            raise NotImplementedError
            print('normalize to 0-255')
            winimg = (winimg * 255).astype('float')
        return winimg
        
    def back_window_transform(self, winimg, width=3000,center=500,normal=False):
        minwindow = float(center)-0.5*float(width)
        if normal:
            winimg = winimg/255
        HUimg = winimg*float(width) + minwindow
        return HUimg


if __name__ == '__main__':
    pass