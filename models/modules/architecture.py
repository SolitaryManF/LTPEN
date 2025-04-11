import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import block as B
from collections import OrderedDict



class Get_ltpe(nn.Module):
    def __init__(self):
        super(Get_ltpe, self).__init__()
        kernel7 = [[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]
        kernel6 = [[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]]
        kernel5 = [[0, 0, -1],
                  [0, 1, 0],
                  [0, 0, 0]]
        kernel4 = [[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]
        kernel3 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]]
        kernel2 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]]
        kernel1 = [[0, 0, 0],
                  [0, 1, 0],
                  [-1, 0, 0]]
        kernel0 = [[0, 0, 0],
                  [-1, 1, 0],
                  [0, 0, 0]]

        kernel7 = torch.cuda.FloatTensor(kernel7).unsqueeze(0).unsqueeze(0)
        kernel6 = torch.cuda.FloatTensor(kernel6).unsqueeze(0).unsqueeze(0)
        kernel5 = torch.cuda.FloatTensor(kernel5).unsqueeze(0).unsqueeze(0)
        kernel4 = torch.cuda.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel3 = torch.cuda.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.cuda.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel1 = torch.cuda.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel0 = torch.cuda.FloatTensor(kernel0).unsqueeze(0).unsqueeze(0)

        self.weight_7 = nn.Parameter(data=kernel7, requires_grad=False)
        self.weight_6 = nn.Parameter(data=kernel6, requires_grad=False)
        self.weight_5 = nn.Parameter(data=kernel5, requires_grad=False)
        self.weight_4 = nn.Parameter(data=kernel4, requires_grad=False)
        self.weight_3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.weight_1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.weight_0 = nn.Parameter(data=kernel0, requires_grad=False)

        weight_list = []
        weight_list.append(self.weight_0),weight_list.append(self.weight_1),weight_list.append(self.weight_2),weight_list.append(self.weight_3)
        weight_list.append(self.weight_4),weight_list.append(self.weight_5),weight_list.append(self.weight_6),weight_list.append(self.weight_7)
        self.weights = weight_list
        self.norm = torch.nn.InstanceNorm2d(1)

    def forward(self, x):

        x_gray = (0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]).unsqueeze(1)
        out = torch.zeros_like(x_gray)

        for j in range(8):
            x_ltpe = F.conv2d(x_gray, self.weights[j], padding=1)
            x_ltpe = (x_ltpe +1)*0.5
            out = out + x_ltpe*(2**j)/255
        out = self.norm(out)
        out = torch.cat((out, out, out), dim=1)

        return out

####################
# Generator
####################

class SPSRNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(SPSRNet, self).__init__()
        feature_channels = nf

        self.get_ltpe = Get_ltpe()

        self.f_HR_conv2 = B.conv_block(out_nc*2, out_nc, kernel_size=3, norm_type=None, act_type=None)

        # 主干参数
        self.conv_1 = conv_layer(in_nc,feature_channels,kernel_size=3)
        self.block_1 = RLFB(feature_channels)
        self.block_2 = RLFB(feature_channels)
        self.block_3 = RLFB(feature_channels)
        self.block_4 = RLFB(feature_channels)
        self.block_5 = RLFB_ori(feature_channels)
        self.block_6 = RLFB_ori(feature_channels)
        self.conv_2 = conv_layer(feature_channels,feature_channels,kernel_size=3)
        self.upsampler = pixelshuffle_block(feature_channels, feature_channels, upscale_factor=upscale)
        self.upsampler_conv = conv_layer(feature_channels, out_nc, kernel_size=3)
        # 纹理分支参数
        self.ltpe_conv_1 = conv_layer(in_nc, feature_channels, kernel_size=3)
        self.ltpe_block_1 = RLFB_woESA(feature_channels)
        self.ltpe_block_2 = RLFB_woESA(feature_channels)
        self.ltpe_block_3 = RLFB_woESA(feature_channels)
        self.ltpe_block_4 = RLFB_woESA(feature_channels)
        self.ltpe_conv_2 = conv_layer(feature_channels, feature_channels, kernel_size=3)
        self.ltpe_upsampler1 = pixelshuffle_block(feature_channels, feature_channels, upscale_factor=upscale)
        self.ltpe_upsampler2 = pixelshuffle_block(feature_channels, feature_channels, upscale_factor=upscale)
        self.ltpe_upsampler1_conv = conv_layer(feature_channels, out_nc, kernel_size=3)
        self.ltpe_upsampler2_conv = conv_layer(feature_channels, out_nc, kernel_size=3)

        self.block_7 = RLFB_ori(feature_channels*2)
        self.out_down = B.conv_block(feature_channels * 2, feature_channels, kernel_size=3, norm_type=None, act_type="leakyrelu")
        self.out_down2 = B.conv_block(feature_channels, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.eca = ECA(channel=3)


    def forward(self, x):    

        x_ltpe = self.get_ltpe(x)

        # 4层RLFB纹理分支
        out_feature_ltpe = self.ltpe_conv_1(x_ltpe)
        out_b1 = self.ltpe_block_1(out_feature_ltpe)
        out_b2 = self.ltpe_block_2(out_b1)
        out_b3 = self.ltpe_block_3(out_b2)
        out_b4_ltpe = self.ltpe_block_4(out_b3)
        out_low_resolution = self.ltpe_conv_2(out_b4_ltpe)
        out_low_resolution_gram = out_low_resolution
        ltpe_output_gram = self.ltpe_upsampler2(out_low_resolution_gram)
        ltpe_output_gram_3 = self.ltpe_upsampler2_conv(ltpe_output_gram)

        # 6层RLFB主干
        out_feature = self.conv_1(x)
        out_b1 = self.block_1(out_feature, out_low_resolution_gram)
        out_b2 = self.block_2(out_b1, out_low_resolution_gram)
        out_b3 = self.block_3(out_b2, out_low_resolution_gram)
        out_b4 = self.block_4(out_b3, out_low_resolution_gram)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)
        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)
        output = self.upsampler_conv(output)

        return ltpe_output_gram_3, output, output, ltpe_output_gram_3


####################
# Perceptual Network
####################

class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value
def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels*2, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels*2, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels*2, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)


    def forward(self, x, ltpe):

        x0 =x
        x = torch.cat([ltpe * 0.2 + x, x], 1)
        out = (self.c1_r(x))
        out = self.act(out)

        out = torch.cat([ltpe * 0.2 + out, out], 1)
        out = (self.c2_r(out))
        out = self.act(out)

        out = torch.cat([ltpe * 0.2 + out, out], 1)
        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x0
        out = self.esa(self.c5(out))

        return out

class RLFB_woESA(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None):
        super(RLFB_woESA, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.c5(out)

        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)

class RLFB_ori(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB_ori, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out