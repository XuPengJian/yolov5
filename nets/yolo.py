import torch
import torch.nn as nn

from nets.CSPdarknet import C3, Conv, CSPDarknet


# yolo_body网络结构的定义,包括CSPdarknet53(SPPBottleneck),FPN结构,YoloHead的整合
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        # 不同大小的模型,在实例化网络时,网络层数和通道数有所不同
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }  # model depth multiple  模型层数倍数
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }  # layer channel multiple  层通道数倍数
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_depth = max(round(dep_mul * 3), 1)  # 3, 模型层数倍数,对应的结果是 {'n': 1, 's': 1, 'm': 2, 'l': 3, 'x': 4, }
        base_channels = int(wid_mul * 64)  # 64, 层通道数倍数,对应的结果是 {'n': 16, 's': 32, 'm': 48, 'l': 64, 'x': 80, }

        # -----------------------------------------------#
        # 输入图片是640, 640, 3
        # 初始的基本通道是64
        # -----------------------------------------------#
        # base_channels(基础通道数)和base_depth(网络深度)是用用于构建网络(实例化)的基础参数
        # phi是版本型号,pretrained是是否采用预训练模型,用于在网络中是否使用预训练模型参数  # TODO:感觉这两个已经没有太大必要了
        self.backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)

        # 上采样模块,上采样到原来的两倍,使用nearest模式
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 对feat3来说,1*1卷积,图片大小不变,通道数变为原来的一半
        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        # 用于feat3上采样(然后与feat2拼接)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        # 对feat2来说,1*1卷积,图片大小不变,通道数变为原来的一半
        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        # 用于feat3上采样与feat2拼接的结果,再上采样,(然后和feat1 拼接)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        # 对feat1来说,downsample下采样环节,通道数不变,但图像缩小一倍(步长为2)
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # C3(CSPlayer)卷积,作用于下采样后的结果
        self.conv3_for_downsample1 = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        # 对feat2来说,downsample下采样环节,通道数不变,但图像缩小一倍(步长为2)
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # C3(CSPlayer)卷积,作用于下采样后的结果
        self.conv3_for_downsample2 = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # yolo检测头部分,基于base_channel和网络结构确定输入的通道数,使用1*1的卷积,控制最终输出通道数
        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes), 拼接了两次, 所以是4倍
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)  # 检测小目标
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes), 拼接了三次, 所以是8倍
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)  # 检测中目标
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes), 拼接了四次, 所以是16倍
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)  # 检测大目标

    # 可以先看forward,再看__init__是如何定义的
    def forward(self, x):
        # 先经过backbone层,得到feat1,feat2,feat3
        feat1, feat2, feat3 = self.backbone(x)

        # FPN结构,整体逻辑为先上采样,再与下采样的结果合并(top-down)自底向上
        # 上采样阶段(其中也要跟feat合并)
        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.conv_for_feat3(feat3)  # 1*1卷积,通道数减半,yolov8取消了这个上采样前的卷积
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)  # 上采样,图片大小变大一倍
        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_upsample, feat2], 1)  # 然后与feat2拼接(拼接之后通道数翻倍)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.conv3_for_upsample1(P4)  # 然后走C3卷积

        # 40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)  # 1*1卷积,通道数减半
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)  # 上采样,图片大小变大一倍
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, feat1], 1)  # 然后与feat1拼接(拼接之后通道数翻倍)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.conv3_for_upsample2(P3)  # 然后走C3卷积

        # PAN结构合并,下采样合并阶段(bottom-up)自顶向下
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)  # 3*3卷积下采样,步长为2,图片缩小一倍
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)  # 然后与P4拼接(拼接之后通道数翻倍)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)  # 然后走C3卷积

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)  # 3*3卷积下采样,步长为2,图片缩小一倍
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)  # 然后与P5拼接(拼接之后通道数翻倍)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)  # 然后走C3卷积

        # ---------------------------------------------------#
        #   第三个特征层:检测小目标
        #   比如y3=(batch_size,75,80,80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # ---------------------------------------------------#
        #   第二个特征层:检测中目标
        #   比如y2=(batch_size,75,40,40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        # ---------------------------------------------------#
        #   第一个特征层:检测大目标
        #   比如y1=(batch_size,75,20,20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        # 最终输出结果为三种尺寸的输出,格式为(batch_size,候选框数量*(5+分类数量),grid_h,grid_w)
        return out0, out1, out2
