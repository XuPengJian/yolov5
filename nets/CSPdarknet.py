import warnings

import torch
import torch.nn as nn


# SiLU激活函数
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# 自动计算padding大小,使得图片size不变
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# CBS结构的定义,用的是SiLU(如果是CBL就是Conv2d+BN+Leaky_relu)
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# 残差模块的一种Res_unit(从初始输入到最终输出,并不改变通道数与图片大小)
# Bottleneck由两个1*1卷积层夹心一个3*3卷积层组成：(building block由两个3*3的卷积层组成)
#   其中1*1卷积层负责减少然后增加（实际上就是恢复）维数，让3*3卷积层成为输入/输出维数更小的瓶颈。
#   Bottleneck既减少了参数量，又优化了计算，保持了原有的精度
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # YOLOv5的Bottleneck相对于resnet来说少了一个1*1 的升维卷积，直接用3*3的卷积还原到c2的维度(其中c1 == c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# CSPlayer结构(从初始输入到最终输出,并不改变通道数与图片大小)
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions(使用了bottleneck并带有三个卷积)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels, e=expansion是用来定义隐藏层维度与输入(=输出)的倍数关系
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        # 对于bottleneck来说,一般c1(input_channel)和c2(output_channel)是保持一致的
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
            , dim=1))


# 类似于空间金字塔池化,实现局部特征和全局特征的featherMap级别的融合(通过concat进行通道拼接,ps:旧版的空间金字塔池化是为了要固定长度的输出,最后接一个全连接层)
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


"""
相当于前面的Backbone主干网络部分(核心部分)
"""


# Backbone主干网络部分
class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth, phi, pretrained):
        """
        :param base_channels: 基础通道数
        :param base_depth: 网络深度层数
        :param phi: 对应的模型版本
        :param pretrained: 是否采用预训练模型
        """
        super().__init__()
        # 输入图片是640, 640, 3
        # 初始的基本通道base_channels是64

        # -----------------------------------------------#
        #   利用卷积结构进行特征提取
        #   640, 640, 3 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = Conv(3, base_channels, 6, 2, 2)

        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.dark2 = nn.Sequential(
            # 320, 320, 64 -> 160, 160, 128
            Conv(base_channels, base_channels * 2, 3, 2),
            # 160, 160, 128 -> 160, 160, 128
            C3(base_channels * 2, base_channels * 2, base_depth),
        )

        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        #                   在这里引出有效特征层80, 80, 256
        #                   进行加强特征提取网络FPN的构建
        # -----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        #                   在这里引出有效特征层40, 40, 512
        #                   进行加强特征提取网络FPN的构建
        # -----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )

        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C3(base_channels * 16, base_channels * 16, base_depth),
            SPPF(base_channels * 16, base_channels * 16),
        )

        # # TODO:此处还需要吗?判断是否为预训练模型,下载backbone网络的预训练模型
        # if pretrained:
        #     backbone = "cspdarknet_" + phi
        #     url = {
        #         "cspdarknet_n": 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_n_v6.1_backbone.pth',
        #         "cspdarknet_s": 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_s_v6.1_backbone.pth',
        #         'cspdarknet_m': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_m_v6.1_backbone.pth',
        #         'cspdarknet_l': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_l_v6.1_backbone.pth',
        #         'cspdarknet_x': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_x_v6.1_backbone.pth',
        #     }[backbone]
        #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
        #     self.load_state_dict(checkpoint, strict=False)
        #     print("Load weights from ", url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # -----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x  # 用于concat,之后用于检测小目标
        # -----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x  # 用于concat,之后用于检测中目标
        # -----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x  # 用于网络继续推理,之后还会有concat合并操作,之后用于检测大目标(感受野更大)
        return feat1, feat2, feat3
