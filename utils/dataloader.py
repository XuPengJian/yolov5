import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

"""
对于每张图片的每个gt，先匹配每个特征层设定的anchor，如果不满足阈值，则该对应的anchor为负样本；然后再匹配特征点，YoloV5是匹配邻近的3个特征点
，通过这样的方式就有多个特征点多个anchor来负责这个gt的预测。

在YoloV5中，训练时正样本的匹配过程可以分为两部分。
a、匹配先验框。
b、匹配特征点。

所谓正样本匹配，就是寻找哪些先验框被认为有对应的真实框，并且负责这个真实框的预测。

a、匹配先验框
在YoloV5网络中，一共设计了9个不同大小的先验框。每个输出的特征层对应3个先验框。

对于任何一个真实框gt，YoloV5不再使用iou进行正样本的匹配，而是直接采用高宽比进行匹配，即使用真实框和9个不同大小的先验框计算宽高比。

如果真实框与某个先验框的宽高比例大于设定阈值，则说明该真实框和该先验框匹配度不够，将该先验框认为是负样本。

比如此时有一个真实框，它的宽高为[200, 200]，是一个正方形。YoloV5默认设置的9个先验框为[10,13], [16,30], [33,23], [30,61], [62,45],
 [59,119], [116,90], [156,198], [373,326]。设定阈值门限为4。
此时我们需要计算该真实框和9个先验框的宽高比例。比较宽高时存在两个情况，一个是真实框的宽高比先验框大，一个是先验框的宽高比真实框大。因此我们需要
同时计算：真实框的宽高/先验框的宽高；先验框的宽高/真实框的宽高。然后在这其中选取最大值。
"""

'''随机生成[a, b)之间的值'''


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


'''将输入图片进行归一化'''


def preprocess_input(image):
    image /= 255.0
    # image -= np.array([0.485, 0.456, 0.406])
    # image /= np.array([0.229, 0.224, 0.225])
    return image


'''
基于txt的逐行数据,读取原始图像image和box数据
'''


def get_data(annotation_line):
    line = annotation_line.split()  # 基于空格和回车划分
    # 读取图像并转换成RGB图像
    image = Image.open(line[0])  # PIL
    image = cvtColor(image)  # 转化为RGB格式,防止灰度图在预测时报错
    # 获得gt框数据,numpy格式[[x1 y1 x2 y2 cls] [...]]获取所有的框,并转换为int格式的numpy格式
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    return image, box


'''将图像转换成RGB图像，防止灰度图在预测时报错'''


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


'''letterbox操作(一般用在验证集和推理阶段,不会用在训练集)'''


def letterbox_resize(image, box, input_shape=(640, 640)):
    # 获得图像的高宽与目标高宽
    iw, ih = image.size  # 图片原始的宽高
    h, w = input_shape  # 输入网络里的宽高

    scale = min(w / iw, h / ih)  # 比例基于长边来确定
    nw = int(iw * scale)  # 按比例缩放
    nh = int(ih * scale)  # 按比例缩放
    # 确定左上角角点的位置,后面用来paste确定位置
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # 将图像多余的部分加上灰条
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32)

    # 对真实框进行调整 [[x1 y1 x2 y2 cls] [...]]
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0  # 有小于零的部分直接设置为0
        box[:, 2][box[:, 2] > w] = w  # x大于w的部分直接设置为w的值
        box[:, 3][box[:, 3] > h] = h  # y大于h的部分直接设置为h的值
        box_w = box[:, 2] - box[:, 0]  # 框的w = x2-x1
        box_h = box[:, 3] - box[:, 1]  # 框的h = y2-y1
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box, 丢弃一些无效的框

    return image_data, box  # 返回处理后的单张图数据和框数据(无随机)


'''
随机高宽比+缩放比例，直接将图片resize到input_shape
'''


def random_resize(image, box, input_shape=(640, 640), jitter=.3, scale_ratio=(.25, 2), random_ar=True):
    """
    :param image: 输入图片需要是PIL.Image格式
    :param box: resize图片的同时也resize box，box的shape应该是[num_anchors, 5(xyxycls)]
    :param input_shape: 模型的输入分辨率大小，这里的缩放是基于这个大小进行的
    :param jitter: 随机高宽比用到的随机偏差值
    :param scale_ratio: 随机缩放的比例范围
    :param random_ar: 是否使用随机高宽比
    :return: 缩放到目标尺寸的image，为PIL.Image格式
    """
    # 获取原图高宽，注意PIL.Image.size得到的是(w, h)
    iw, ih = image.size
    # 获取目标高宽比
    h, w = input_shape
    # 获得随机缩放比例
    scale_min, scale_max = scale_ratio
    scale = rand(scale_min, scale_max)
    if random_ar:
        # 获得随机宽高比
        new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        # 如果宽高比小于1，则先用高乘以缩放比例，然后用变换后的高乘以宽高比得到宽
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        # 如果宽高比大于等于1，则先用宽乘以缩放比例，然后用变换后的宽除以宽高比得到高
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
    else:
        nh = int(scale * h)
        nw = int(scale * w)
    # 缩放至变换后的宽高  TODO:也许可以用随机的缩放模式
    image = image.resize((nw, nh), Image.BICUBIC)
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image
    # 对真实框进行调整 [[x1 y1 x2 y2] [...]]
    if len(box) > 0:
        np.random.shuffle(box)  # 随机打乱box的顺序
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx  # 对box缩放偏移到正确的位置
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy  # 对box缩放偏移到正确的位置
        box[:, 0:2][box[:, 0:2] < 0] = 0  # 有小于零的部分直接设置为0
        box[:, 2][box[:, 2] > w] = w  # x大于w的部分直接设置为w的值
        box[:, 3][box[:, 3] > h] = h  # y大于h的部分直接设置为h的值
        box_w = box[:, 2] - box[:, 0]  # 框的w = x2-x1
        box_h = box[:, 3] - box[:, 1]  # 框的h = y2-y1
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box, 丢弃一些无效的框

    return image, box


'''
随机水平翻转
'''


def random_horizontal_flip(image, box, input_shape=(640, 640), threshold=0.5):
    h, w = input_shape
    v = random.random()
    if v < threshold:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(box) != 0:
            box[:, [0, 2]] = w - box[:, [2, 0]]
    return image, box


'''
随机竖直翻转
'''


def random_vertical_flip(image, box, input_shape=(640, 640), threshold=0.5):
    h, w = input_shape
    v = random.random()
    if v < threshold:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if len(box) != 0:
            box[:, [1, 3]] = h - box[:, [1, 3]]
    return image, box


'''
随机HSV增强
'''


def random_hsv(image, hue=0.1, sat=0.7, val=0.4):
    """
    :param image: 输入的图片，需要为numpy.array格式
    :param hue: H的随机扰动比例
    :param sat: S的随机扰动比例
    :param val: V的随机扰动比例
    :return: image: numpy.array
    """
    # 将图片从RGB转换为HSV，好处是通过变换S和V可以改变亮度和饱和度，做到数据增强
    # 随机取三个[-1, 1)的值，乘以输入的[hgain, sgain, vgain]再加1，这里获取的是三个1左右的比值
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    # 将图像转到HSV上
    # H表示色调，取值范围是[0,180]
    # S表示饱和度，取值范围是[0,255]
    # V表示亮度，取值范围是[0,255]
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype = image.dtype
    # 应用变换
    # x=[0, 1 ... 255]
    x = np.arange(0, 256, dtype=r.dtype)
    # 对H值添加扰动，这个扰动一般较小，H值最大为180，所以要对180取余
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    # 对S值添加扰动
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    # 对V值添加扰动
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    # cv2.LUT(src, lut, dst=None) LUT是look-up table(查找表)的意思
    # src：输入的数据array，类型为8位整型（np.uin8)
    # lut：查找表
    # 这里cv2.LUT(hue, lut_hue)的作用就是将hue原来的值作为索引去lut_hue中找到对应的新值，然后赋给hue
    # 比如在hue中有个值是100，则取lut_hue[100]作为hue当前位置的新值
    # cv2.merge:合并通道，不用指定维度，但是这个操作比较耗时，所以改用np.stack
    # image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = np.stack((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)), axis=2)
    # HSV --> RGB
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
    return image_data


'''
mosaic数据增强(包含各个图片random的步骤)
'''


def get_random_data_with_Mosaic(annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
    h, w = input_shape
    min_offset_x = rand(0.3, 0.7)
    min_offset_y = rand(0.3, 0.7)

    image_datas = []  # 用于储存处理后的图片
    box_datas = []  # 用于储存处理后的框的信息
    index = 0  # 图片序号0-3变化

    # 遍历每一张图片
    for line in annotation_line:
        image, box = get_data(line)
        # 原始图片大小,转化为高宽
        iw, ih = image.size
        image_shape = (ih, iw)

        # 数据增强:给每一张mosaic的图片使用数据增强
        # 随机水平翻转(注意:这里用的是原始图片的input_shape,即image_shape)
        image, box = random_horizontal_flip(image, box, image_shape)
        # 随机垂直翻转(注意:这里用的是原始图片的input_shape,即image_shape)
        image, box = random_vertical_flip(image, box, image_shape)

        # 数据增强:对图像进行缩放并且进行长和宽的扭曲
        # jitter 扰动的比例
        new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)  # 随机扰动一定的宽高比
        scale = rand(.4, 1)  # 图像随机缩放比例
        if new_ar < 1:  # 如果 w/h<1 那么以h为基准缩放
            nh = int(scale * h)  # 随机缩放比例
            nw = int(nh * new_ar)  # 代入高宽比
        else:  # 如果 w/h>1 那么以w为基准缩放
            nw = int(scale * w)  # 随机缩放比例
            nh = int(nw / new_ar)  # 代入高宽比
        image = image.resize((nw, nh), Image.BICUBIC)  # 将图片缩放成指定尺寸

        # 将图片进行定位，分别对应四张分割图片的位置
        # 0 3
        # 1 2
        # min_offset_x和min_offset_y是对于hw划分四个区域随机中心点的位置
        if index == 0:
            dx = int(w * min_offset_x) - nw
            dy = int(h * min_offset_y) - nh
        elif index == 1:
            dx = int(w * min_offset_x) - nw
            dy = int(h * min_offset_y)
        elif index == 2:
            dx = int(w * min_offset_x)
            dy = int(h * min_offset_y)
        elif index == 3:
            dx = int(w * min_offset_x)
            dy = int(h * min_offset_y) - nh

        # 创建一个size为(h,w)的新的三维度灰度图像,作为黏贴用的底板(每次循环都会创建一个新的灰图,所以之后还要再将这分开生成的四张图再切分拼在一起)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))  # 将一张图片粘贴到另一张图片,设置的是图片左上角的位置
        image_data = np.array(new_image)  # 转化为numpy格式

        index = index + 1  # 图片序号+1
        box_data = []  # 储存单张图片的框信息

        # 对box的数值进行重新映射处理(numpy格式)
        if len(box) > 0:
            np.random.shuffle(box)  # 对box进行打乱[[x1 y1 x2 y2 cls][...]]
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx  # 对x坐标重新映射
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy  # 对y坐标重新映射
            box[:, 0:2][box[:, 0:2] < 0] = 0  # 映射完后,左上角(x1,y1)小于零的部分直接给0(截取掉)
            box[:, 2][box[:, 2] > w] = w  # x2大于w的部分直接取w
            box[:, 3][box[:, 3] > h] = h  # y2大于h的部分直接取h
            box_w = box[:, 2] - box[:, 0]  # 计算出宽度数据 w = x2 - x1
            box_h = box[:, 3] - box[:, 1]  # 计算出高度数据 h = y2 - y1
            # 求得box_w和box_h都大于1的部分,并生成掩码矩阵,并筛选符合要求的box框,排除掉那些框已经被裁没的
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))  # 生成一个空白的矩阵,shape为(len(box),5),5代表x,y,w,h,cls
            box_data[:len(box)] = box  # 将box的值赋予给box_data

        image_datas.append(image_data)  # 将处理好的4张图片数据加入到列表中(现在还在各自的图片中,还没拼在一起)
        box_datas.append(box_data)  # 将处理好的框数据加入到列表中

    # 一次循环结束后(四张图),再将图片分割，拼在一起(基于之前定义的中心点,之前是分开生成的,只是定好了位置)
    cutx = int(w * min_offset_x)
    cuty = int(h * min_offset_y)

    # 创建一张三维度的,hw的黑图片
    # 0 3
    # 1 2
    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]  # 图片0
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]  # 图片1
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]  # 图片2
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]  # 图片3
    # 将图片转化为 np的 uint8格式
    new_image = np.array(new_image, np.uint8)

    # 拼完之后做数据增强, 对图像进行色域变换,计算色域变换的参数
    new_image = random_hsv(new_image, hue, sat, val)

    # 对框进行进一步处理, 输入框数据和切割的位置
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes  # 返回mosaic处理后整合好的图片和框数据


'''
合并框数据
'''


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []  # 临时储存框
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # 对于序号为1的框来说,处理一些超出边界或贴边的情况
            if i == 0:
                if y1 > cuty or x1 > cutx:  # 超出范围直接取下一个框(不在范围内)
                    continue
                if y2 >= cuty and y1 <= cuty:  # y1在范围内,y2超出范围,则y2的值直接取cuty
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:  # x1在范围内,x2超出范围,则x2的值直接取cutx
                    x2 = cutx

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
            # 将处理后的框信息合并到tmp_box中
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            # 将完整的临时框放入merge_bbox中
            merge_bbox.append(tmp_box)
    return merge_bbox


'''
mixup数据增强
'''


def random_mixup(image_1, box_1, image_2, box_2):
    # mixup操作,颜色各取一半,混合加起来
    new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
    # 将box_1和box_2的框数据合并(之后在计算loss的时候每个物体的loss按照mixup的系数进行加权求和)
    if len(box_1) == 0:
        new_boxes = box_2
    elif len(box_2) == 0:
        new_boxes = box_1
    else:
        new_boxes = np.concatenate([box_1, box_2], axis=0)
    return new_image, new_boxes


"""
YOLOv5 Dataset处理部分
"""


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length,
                 mosaic, mixup, mosaic_prob, mixup_prob, train=True, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio  # 使用数据增强所占epoch的比例

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

        self.bbox_attrs = 5 + num_classes
        self.threshold = 4  # gt和候选框差距最大的边比值的阈值

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length  # 这里用index%length,防止index out of range
        # 训练时进行数据的随机增强
        # 验证时不进行数据的随机增强
        # 这里使用mosaic数据增强
        # 这里的epoch_now,是会在每个epoch里面进行更新的,用来决定采用mosaic的数据增强的范围

        # 训练集数据增强处理
        if self.train:
            if self.mosaic and random.random() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
                lines = random.sample(self.annotation_lines, 3)  # 从序列seq(训练集)中选择n个随机且独立的三个图片
                lines.append(self.annotation_lines[index])  # 再拼接一个当前index上的图片
                random.shuffle(lines)  # 对输入的四张图片图片路径进行打乱(后面用来做mosaic)
                image, box = get_random_data_with_Mosaic(lines, self.input_shape)
                # 这里使用mixup数据增强(是基于mosaic数据增强的进一步处理)
                if self.mixup and random.random() < self.mixup_prob:
                    lines = random.sample(self.annotation_lines, 1)  # 从所有的数据中随机采样一条图片数据
                    # image_2, box_2是用于mixup的一张大图
                    # ----------
                    # 数据增强部分
                    # ----------
                    # 获取原始图像数据和框数据
                    image_2, box_2 = get_data(lines[0])
                    # 随机缩放(包括随机长宽比),缩放到input_shape大小
                    image_2, box_2 = random_resize(image_2, box_2, self.input_shape, jitter=.3, scale_ratio=(.25, 2))
                    # 随机水平翻转
                    image_2, box_2 = random_horizontal_flip(image_2, box_2, self.input_shape)
                    # 转为为numpy格式的uint8
                    image_2 = np.array(image_2, np.uint8)
                    # 随机hsv数据增强
                    image_2 = random_hsv(image_2, hue=.1, sat=0.7, val=0.4)
                    # mixup合并image和image_2
                    image, box = random_mixup(image, box, image_2, box_2)
            # 否则,使用普通的数据增强就好(训练集部分)
            else:
                # ----------
                # 数据增强部分
                # ----------
                # 获取原始图像数据和框数据
                image, box = get_data(self.annotation_lines[index])
                # 随机缩放(包括随机长宽比),缩放到input_shape大小
                image, box = random_resize(image, box, self.input_shape, jitter=.3, scale_ratio=(.25, 2))
                # 随机水平翻转
                image, box = random_horizontal_flip(image, box, self.input_shape)
                # 随机垂直翻转
                image, box = random_vertical_flip(image, box, self.input_shape)
                # 转为为numpy格式的uint8
                image = np.array(image, np.uint8)
                # 随机hsv数据增强
                image = random_hsv(image, hue=.1, sat=0.7, val=0.4)

        # 验证集的处理(验证集不使用数据增强)
        else:
            image, box = get_data(self.annotation_lines[index])
            image, box = letterbox_resize(image, box, self.input_shape)

        # 归一化操作
        # 将图片img归一化在0到1之间,并从[H W C]->[C H W],并将数据都转变为numpy的np.float32
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            # 对真实框进行归一化，调整到0-1之间 [[x1 y1 x2 y2 cls] [...]]
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]  # x/input_w
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]  # y/input_h
            # 序号为0、1的部分，为真实框的中心
            # 序号为2、3的部分，为真实框的宽高
            # 序号为4的部分，为真实框的种类
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]  # 求真实框宽高 w = x2-x1, h = y2-y1
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2  # 求真实框中心点 x = x1+w/2, y = y1+h/2

        # 得到y_true
        y_true = self.get_target(box)  # box的格式为[[x y w h cls] [...]],并且是归一化x,y,w,h到0-1之间.外面的总数是代表每张图框的数量num_true_box
        # y_true的内容为[[3(anchor_num),20*20,5+cls],[3(anchor_num),40*40,5+cls],[3(anchor_num),60*60,5+cls]]
        return image, box, y_true  # 这里的 box其实就是target,格式为(x y a h cls)

    """
    在过去的Yolo系列中，每个真实框由其中心点所在的网格内的左上角特征点来负责预测。对于被选中的特征层，首先计算真实框落在哪个网格内，此时该网格
    左上角特征点便是一个负责预测的特征点。
    同时利用四舍五入规则，找出最近的两个网格，将这三个网格都认为是负责预测该真实框的。
    """

    # TODO:这里没有考虑左上右上左下右下,是否存在一点问题呢?
    # 以0.5为划分通过sub_x与sub_y(网格的相对位置),增加多相邻的两个网格作为y_true.据说是为了扩充正样本,但可能会对小目标有影响
    # 匹配邻近的3个特征点
    def get_near_points(self, x, y, i, j):
        # (i,j)代表格子左上角的点,(x,y)是对应具体位置的点
        sub_x = x - i  # 获得x差距值
        sub_y = y - j  # 获得y差距值
        # 如果差值都大于0.5,则都向外偏移1
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        # 如果x的差值小于0.5,y的差值大于0.5,则向x偏移-1向y偏移1
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        # 如果如果差值都小于0.5,则都向内偏移1
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        # 如果x的差值大于0.5,y的差值小于0.5,则向x偏移1向x偏移-1
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, targets):
        # targets传入的是box的格式为[[x y w h cls] [...]],并且是归一化x,y,w,h到0-1之间,为了得到y_true的数据,之后用来算loss
        # 此方法是针对一张图用的,一张图可以有很多个框
        # 一共有三个特征层数
        num_layers = len(self.anchors_mask)  # 计算mask的数量,相当于特征层的数量,比如anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = np.array(self.input_shape, dtype='int32')  # 将input_shape改为numpy格式,[h w]比如input_shape = [640 640]

        # 计算每个grid的shape大小(依次为大目标,中目标,小目标),{3:4}一般用不太到,只有三个特征层,0,1,2. 比如grid_shapes = [[20 20] [40 40] [80 80]]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8, 3: 4}[l] for l in range(num_layers)]  # l代表layer

        # 所以y_true也要根据特征层数量分三层创建,创建三个零矩阵作为模板.
        # 比如y_true=[np.zeros(3, 20, 20, 5+80),np.zeros(3, 40, 40, 5+80),np.zeros(3, 80, 80, 5+80)]
        y_true = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs),
                           dtype='float32') for l in range(num_layers)]

        # 比如box_best_ratio = [np.zeros(3, 20, 20),np.zeros(3, 40, 40),np.zeros(3, 80, 80)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32')
                          for l in range(num_layers)]

        # 如果targets为空,直接返回y_ture,全是0
        if len(targets) == 0:
            return y_true

        # 基于特征层数量进行循环
        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]  # 获取该特征层的grid_shape的高宽尺寸,比如:in_h=20,in_w=20
            # 读取txt的候选框大小,这里共有9个框,shape为(9,2),2记录的是w,h信息,并除以特征层的grid_size的大小,比如除以32
            anchors = np.array(self.anchors) / {0: 32, 1: 16, 2: 8, 3: 4}[l]

            batch_target = np.zeros_like(targets)  # 构造与targets维度相同的矩阵,并初始化为全0
            # 计算出正样本在特征层上的中心点,重新映射到in_h*in_w的特征图上的大小
            # batch_target的格式为[[x y w h cls] [...]],并将gt的值映射上去
            batch_target[:, [0, 2]] = targets[:, [0, 2]] * in_w  # x和w
            batch_target[:, [1, 3]] = targets[:, [1, 3]] * in_h  # y和h
            batch_target[:, 4] = targets[:, 4]  # cls
            # -------------------------------------------------------#
            #   用于增加维度
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #
            #   这里利用了python的广播机制
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            # -------------------------------------------------------#
            # wh和先验框计算比值
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            # 沿着最后一个维度(2个w的比值,h的比值)进行拼接(信息不会丢失,得到4个w的比值,h的比值)
            ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
            # 在最后一个维度求取宽高的比值中的最大值(得到差异最大的边的最大值,这里取最大值顺便把值都限制在>1) shape为num_true_box, 9
            max_ratios = np.max(ratios, axis=-1)

            # 遍历单张图每个框num_true_box计算出来的最大值(w,h,1/w,1/h的比值中的最大值)
            for t, ratio in enumerate(max_ratios):
                # t:num_true_box 代表第几个框, ratio的长度是9  比如[1 2 3 4 5 6 7 8 9]

                # 为了得到掩码(筛选出符合阈值范围内的候选框)
                # 阈值设置,超过threshold的设置为False,没超过threshold的都设置为True,比如这里的threshold设置为4,比如只有1,2,3 位置的会设置为True
                over_threshold = ratio < self.threshold  # 得到掩码矩阵
                # 其中最小值的部分设置为True,即最符合候选框的最好的比例(即使超过了阈值,每个框至少也要选一个最小的,最接近的,不然就丢失信息了)
                over_threshold[np.argmin(ratio)] = True  # 其中最小值也设置为True

                # 遍历某个特征层的所有anchor_mask,比如anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] l=0中的 [6, 7, 8]
                # 可以理解成获取属于这一层的候选框,一层一层,一组一组来遍历
                for k, mask in enumerate(self.anchors_mask[l]):
                    # 此处过滤掉不满足threshold的框:比如如果mask[6,7,8]中的6在over_threshold中False,则执行continue
                    # 继续依次循环找下一个框[[7,8]],如果都没有,那么就等着下个特征层(num_layers)的循环再选出符合阈值/最小值要求的候选框
                    if not over_threshold[mask]:
                        continue
                    # 获得真实框对应的网格点(左上角的交点)
                    # x  1.25     => 1
                    # y  3.75     => 3
                    # t 代表第t个框
                    i = int(np.floor(batch_target[t, 0]))  # x
                    j = int(np.floor(batch_target[t, 1]))  # y

                    # 获取一个偏移值,基于实际点在grid的具体位置,得到轻微偏移的local位置(匹配邻近的3个特征点,据说是为了扩充正样本的数量)
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]
                        # 如果计算得到的新的local_i或者local_j超出了grid_shape范围,则执行continue重新循环,继续找新的local点
                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue
                        # box_best_ratio的shape为(3,20,20)或(3,40,40)或(3,80,80)的全为0的数组,3是每个特征层对应框的数量
                        # 对gt中心点周围的grid进行赋值
                        # 如果当前位置比例最好的框已经不为0(说明y_true也已经赋上了值)
                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            # 如果当前最好的比例比ratio[mask]还要大(说明不是最好的框),那么把之前给的y_true值设置为0
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                # 那么y_true对应的值重新设置为0
                                y_true[l][k, local_j, local_i, :] = 0
                            # 如果当前最好比例的框已经是最好的,并且y_true已经附上了相应的值,直接跳过,进行下一轮循环
                            else:
                                continue

                        # 取出真实框的种类(用来后面独热编码的方式显示)
                        c = int(batch_target[t, 4])  # 这里是用类别序号表示的

                        # 给特定位置给具体值
                        # k为框的序号
                        # tx、ty代表中心调整参数的真实值(原始值)
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]  # x
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]  # y
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]  # w
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]  # h
                        y_true[l][k, local_j, local_i, 4] = 1  # 赋予置信度(只要存在该类别)
                        y_true[l][k, local_j, local_i, c + 5] = 1  # 类别(独热形式表现,+5是为了让序号往后移)
                        # 获得当前先验框最好的比例,存储在box_best_ratio中(是用来做判断使用)
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]

        # 这里的y_true是用来之后跟预测值算loss使用的, y_true是一个长度为3(特征层数量的列表),
        # y_true的内容为[[3(anchor_num),20*20,5+cls],[3(anchor_num),40*40,5+cls],[3(anchor_num),60*60,5+cls]]
        return y_true


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    y_trues = [[] for _ in batch[0][2]]
    for img, box, y_true in batch:
        images.append(img)
        bboxes.append(box)
        for i, sub_y_true in enumerate(y_true):
            y_trues[i].append(sub_y_true)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
    return images, bboxes, y_trues
