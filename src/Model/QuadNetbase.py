from Model.Basic_FFC import *


class FFC_Bottle(nn.Module):
    ''''''
    def __init__(self,n_downsampling, max_features=1024, ngf=64, 
                n_blocks=6, padding_type='reflect', inline = False,
                activation_layer='ReLU', norm_layer=nn.BatchNorm2d, spatial_transform_layers=None, spatial_transform_kwargs=None,resnet_conv_kwargs=None):
        super(FFC_Bottle, self).__init__()
        
        model = []
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        ### resnet Fourier blocks
        # if resnet_conv_kwargs['ratio_gin'] == 0: # [DOING]
        #     print('inline is True')
        #     inline = False
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck,             
                                        padding_type=padding_type, activation_layer=activation_layer,
                                        inline = inline, norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]
        self.ffc_resnet = nn.Sequential(*model)

    def forward(self,x):
        return self.ffc_resnet(x)


class FFCResNetGenerator(nn. Module):
    '''
    LAMA generator
    '''
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn. BatchNorm2d,
                 padding_type='reflect', activation_layer=nn. ReLU,
                 up_norm_layer=nn. BatchNorm2d, up_activation=nn. ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super(). __init__()
        
        model = [nn. ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs. get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        for i in range(n_downsampling): #3
            mult = 2 ** (n_downsampling - i)
            model += [nn. ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn. ReflectionPad2d(3),
                  nn. Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self. model = nn. Sequential(*model)

    def forward(self, input):
        # print(self.model)
        return self. model(input)


class Sino_down(nn.Module):
    def __init__(self,n_downsampling,downsample_conv_kwargs,resnet_conv_kwargs,
                max_features=1024,ngf=64,
                norm_layer=nn. BatchNorm2d,activation_layer=nn. ReLU):
        super(Sino_down, self).__init__()
        
        model = []
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs. get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
                # min in channel out channel,最大不超过，写的太麻烦了
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]
        self.down = nn.Sequential(*model)
        # print(self.down)
    def forward(self,x):
        return self.down(x)


class Sino_Up(nn.Module):
    def __init__(self,n_downsampling,max_features=1024,ngf=64,
                                    up_norm_layer=nn. BatchNorm2d, up_activation=nn. ReLU(True),):
        super(Sino_Up, self).__init__()
        model = []
        for i in range(n_downsampling): #3
            mult = 2 ** (n_downsampling - i)
            model += [nn. ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]
        self.up = nn.Sequential(*model)
        print(self.up)
    def forward(self,x):
        return self.up(x)
    
class Fourier_bottle(nn.Module):
    def __init__(self,n_downsampling,max_features=1024,ngf=64,n_blocks=6,padding_type='reflect',
                 activation_layer='ReLU',norm_layer=nn. BatchNorm2d,spatial_transform_layers=None,spatial_transform_kwargs=None,):
        super(Fourier_bottle, self).__init__()
        model = []
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        ### resnet Fourier blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]
        model += [ConcatTupleLayer()]
        self.bottle_inpainting = nn.Sequential(*model)

    def forward(self,x):
        return self.bottle_inpainting(x)

class Sinogram_FFC_generator(nn. Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=6, norm_layer=nn. BatchNorm2d,
                 padding_type='reflect', activation_layer=nn. ReLU,
                 up_norm_layer=nn. BatchNorm2d, up_activation=nn. ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super(). __init__()
        self.reflect = nn.Sequential(
                nn. ReflectionPad2d(3),
                FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, actinorm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)
        )
        #downsample
        # self.downsample = Sino_down(n_downsampling,downsample_conv_kwargs,
        #                         max_features,ngf, norm_layer,activation_layer)
        self.downsample = Sino_down(n_downsampling,downsample_conv_kwargs,resnet_conv_kwargs)
        #Fourier inpainting
        model = []
        self.bottle_inpainting = Fourier_bottle(n_downsampling,max_features,ngf,n_blocks,padding_type,
                 activation_layer,norm_layer,spatial_transform_layers,spatial_transform_kwargs)

        ### upsample
        self.upsample = Sino_Up(n_downsampling,max_features,ngf,
                                    up_norm_layer, up_activation)
        ## final_process
        model = []
        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn. ReflectionPad2d(3),
                  nn. Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model. append(get_activation('tanh' if add_out_act is True else add_out_act))
        self. final_process = nn. Sequential(*model)


    def forward(self, input):
        x = input
        x = self.reflect(x)
        x = self.downsample(x)
        # _a,_b = x
        # print(type(_a),type(_b))
        # print(_a.shape)
        # print(_b.shape)
        x = self.bottle_inpainting(x)
        print(x.shape)
        x = self.upsample(x)
        print(x.shape)
        output = self.final_process(x)
        # print(self.model)
        return output


# FFC encoder decoder

class ToTupleLayer(nn. Module):
    def __init__(self, ratio_gin,):
        super().__init__()
        self.ratio_gin = ratio_gin
    def forward(self, x):
        assert torch.is_tensor(x)
        _, c, _, _ = x.size()
        if self.ratio_gin != 0:
            x_l, x_g = x[:, : -int(self.ratio_gin * c)], x[:, : int(self.ratio_gin * c)]
        else:
            x_l, x_g = x, 0
        # print(x_l.shape, x_g.shape)
        out = x_l, x_g
        return out


class Encoder_Down(nn.Module):
    def __init__(self, n_downsampling, ngf=64, max_features=1024,
                norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU, resnet_conv_kwargs={}, downsample_conv_kwargs={}):
        super(Encoder_Down, self).__init__()
        
        model = []
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]
        self.down = nn.Sequential(*model)
        
    def forward(self,x):
        return self.down(x)


class Decoder_Up(nn.Module):
    '''提供任意奇数的upsample'''
    def __init__(self,n_downsampling,max_features=1024,ngf=64,
                up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),):
        super(Decoder_Up, self).__init__()

        model = []
        for i in range(n_downsampling): #3
            mult = 2 ** (n_downsampling - i)
            model += [nn. ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]
        self.up = nn.Sequential(*model)

    def forward(self,x):
        return self.up(x)

class FFC_Encoder(nn. Module):
    '''
    [support odd input]
    '''
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=6, 
                norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU,
                init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                add_out_act=True, max_features=1024, ):
        assert (n_blocks >= 0)
        super(). __init__()
        self.reflect = nn.Sequential(
                nn. ReflectionPad2d(3),
                FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs))
        # downsample
        self.downsample = Encoder_Down(n_downsampling, ngf, max_features, 
                    norm_layer, activation_layer, 
                    resnet_conv_kwargs=resnet_conv_kwargs, downsample_conv_kwargs=downsample_conv_kwargs)
        
        # ffc resblock
        self.ffc_bottle = FFC_Bottle(n_downsampling, ngf = ngf, max_features = max_features,
                 n_blocks = n_blocks, padding_type = padding_type, activation_layer =  activation_layer, norm_layer = norm_layer,
                resnet_conv_kwargs = resnet_conv_kwargs)
        self.cattuple_layer = ConcatTupleLayer()

    def forward(self, x):
        x = self.reflect(x)
        x = self.downsample(x)
        x = self.ffc_bottle(x)
        latent_x = self.cattuple_layer(x)
        return latent_x

    
class FFC_Decoder(nn. Module):
    '''
    FFC decoder
    '''
    def __init__(self, input_nc, output_nc, n_downsampling=3, n_blocks=6, 
                ngf=64, max_features=1024, 
                norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                resnet_conv_kwargs={}, add_out_act=True, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super(). __init__()

        # decoder ffc resblock [latent_x -> decoder ffc_block]
        self. totuple_layer =  ToTupleLayer(resnet_conv_kwargs.get('ratio_gin', 0))
        # self.ffc_bottle = FFC_Bottle(n_downsampling, max_features = max_features,
        #         ngf = ngf, n_blocks = n_blocks//2, padding_type = padding_type, activation_layer =  activation_layer, norm_layer = norm_layer, inline = False, resnet_conv_kwargs = resnet_conv_kwargs)
        self.ffc_bottle = FFC_Bottle(n_downsampling, max_features = max_features,
                ngf = ngf, n_blocks = n_blocks, padding_type = padding_type, activation_layer =  activation_layer, norm_layer = norm_layer, inline = False, resnet_conv_kwargs = resnet_conv_kwargs)

        # upsample
        self.cattuple_layer = ConcatTupleLayer()
        self.upsample = Decoder_Up(n_downsampling,max_features,ngf,
                                    up_norm_layer, up_activation)
        # final_process
        model = []
        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                    norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]
        model += [nn. ReflectionPad2d(3),
                nn. Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model. append(get_activation('sigmoid' if add_out_act is True else add_out_act))
        self. final_process = nn. Sequential(*model)

    def forward(self, x):
        x = self.totuple_layer(x)
        x = self.ffc_bottle(x)
        x = self.cattuple_layer(x)
        x = self.upsample(x)
        y = self.final_process(x)
        return y



# image domain
class FFC_t(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC_t, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin) 
        in_cl = in_channels - in_cg 
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.lin_c = in_cl
        self.gin_c = in_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        # print('++++++', in_cl,out_cl,in_cg,out_cg)

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        
        # 1*1卷积
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d 
        self.gate = module(in_channels, 2, 1) 

    def forward(self, x):
        x_l, x_g = torch.split(x, [self.lin_c,self.gin_c],dim=1)
        out_xl, out_xg = 0, 0
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            if x_g.size(1) == 0:
                out_xl = self.convl2l(x_l)
            else:
                out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            if x_l.size(1) == 0:
                out_xg = self.convg2g(x_g)
            else:
                out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg

class FFC_BN_ACT_t(nn.Module):
    '''
    Fourier unet skip connection
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=1, ratio_gout=1,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT_t, self).__init__()
        self.ffc = FFC_t(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        # print(x_l.shape,x_g.shape)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        if type(x_l)==int:
            return x_g
        elif type(x_g)==int:
            return x_l
        # return x_l, x_g
        return torch.cat((x_l,x_g),dim=1)



# ================ skip connection ===============
class Fourier_Connection(nn.Module):
    '''Fourier skip connection implement
    1.Fourier connection
    2.Fourier residual skip connection
    3.F + conv3
    4.F + conv3 + residual
    5.F filter net: group = inchannel = outchannel
    '''
    def __init__(self, 
                in_channels,
                out_channels,
                norm_layer=nn.Identity, 
                activation_layer=nn.Identity,):
        super(Fourier_Connection, self).__init__()
        self._f_block = FourierUnit(in_channels, out_channels)
        self._spectralTransform = SpectralTransform(in_channels, out_channels,
                                                enable_lfu = False)
        
        # vanilla conv block
        self._vanilla_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # BFN style: GN instead of BN
        self._bfn_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups = out_channels//16, 
                        num_channels = out_channels),
            nn.ReLU(inplace=True),
        )
        self._identity = nn.Identity(inplace=True)
        self._norm = norm_layer(inplace=True)
        self._act = activation_layer(inplace=True)
    
    def forward(self, x):
        raise Exception("Fourier connection not implemented")


class Fourier_Skip_Block(Fourier_Connection):
    '''Fourier_block + residual
    parm:
        F_block_select: f_unit--FourierUnit s_unit--SpectralTransform 
        C_block_select: identity--Identity vanilla--vanilla bfn--GN style
        residual: True False
    '''
    def __init__(self,
                in_channels,
                out_channels,
                norm_layer=nn.Identity, 
                activation_layer=nn.Identity,
                F_block_select= 'f_unit',
                C_block_select = 'identity',
                residual = True,
                ): 
        super(Fourier_Skip_Block, self).__init__(in_channels,
                                                    out_channels,
                                                    norm_layer=norm_layer, 
                                                    activation_layer=activation_layer)
        assert F_block_select=='f_unit' or F_block_select=='s_unit' or F_block_select=='identity'
        assert C_block_select=='identity' or C_block_select=='vanilla' or C_block_select=='bfn'
        
        # F_block: fourier C_block:conv residual
        if F_block_select == 'f_unit':
            self.F_block = self._f_block
        elif F_block_select == 's_unit':
            self.F_block = self._spectralTransform
        elif F_block_select == 'identity':
            self.F_block = self._identity
        else:
            raise Exception('Fourier_Residual_Skip F_block select fail')
        
        # C_blcok: conv block
        if C_block_select == 'identity':
            self.C_block = self._identity
        elif C_block_select == 'vanilla':
            self.C_block = self._vanilla_conv
        elif C_block_select == 'bfn': 
            self.C_block = self._bfn_conv
        else:
            raise Exception('Fourier_Residual_Skip C_block select fail')

        self.residual = residual 
        self.norm = self._norm
        self.act = self._act

    def forward(self, x):
        f_x = self.F_block(x)
        c_x = self.C_block(x)
        if self.residual:
            x_o = x + f_x + c_x
        else:
            x_o = f_x + c_x
        # log output
        x_o = self.norm(x_o)
        x_o = self.act(x_o)
        output = x_o
        return output 








if __name__ == '__main__':
    # usage
    init_conv_kwargs = {'ratio_gin':0,'ratio_gout':0,'enable_lfu':False}
    downsample_conv_kwargs = {'ratio_gin':0,
                              'ratio_gout':0,
                              'enable_lfu':False}

    resnet_conv_kwargs = {'ratio_gin':0.75,
                            'ratio_gout':0.75,
                            'enable_lfu':False}

    FCres = Sinogram_FFC_generator(input_nc = 2, output_nc = 1,add_out_act='sigmoid', n_downsampling=2,
                                init_conv_kwargs=init_conv_kwargs,
                                downsample_conv_kwargs = downsample_conv_kwargs,
                                resnet_conv_kwargs=resnet_conv_kwargs,)
    device = 'cuda:6'
    input = torch.randn(4,2,640,640)
    output = FCres(input)
    print(input.shape,output.shape)
