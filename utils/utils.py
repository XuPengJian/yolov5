import os
import colorsys
import numpy as np
from PIL import Image

import torch
import cv2

'''
将图像转换成RGB图像，防止灰度图在预测时报错,可以读PIL和numpy格式。
代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
'''


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


'''
对输入图像进行resize
'''


def resize_image(image, size, letterbox_image):
    iw, ih = image.size  # 原图尺寸
    w, h = size  # 目标尺寸
    # 如果为letterbox_image模式
    if letterbox_image:
        scale = min(w / iw, h / ih)  # 选取需要乘的倍数最小的那个(不管是大变小还是小变大,都是基于长边来确定)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))  # 创建一个灰色空白图像
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 把resize后的图像paste上去
    # 直接resize
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


'''
获得类
'''


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


'''
先验框
'''


# 从文件中读取先验框的数据(w,h),这个框的大小是基于input_shape的图像尺寸来计算的
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)  # 两个为一组进行设置
    return anchors, len(anchors)  # 数据格式为[[w1 h1] [w2 h2] ...[w9 h9]] 共9个(默认从小到大排序)


'''
获取学习率
'''


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 归一化图片数据
def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


'''
下载预训练权重
'''


def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    backbone = "cspdarknet_" + phi
    download_urls = {
        "cspdarknet_n": 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_n_v6.1_backbone.pth',
        "cspdarknet_s": 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_s_v6.1_backbone.pth',
        'cspdarknet_m': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_m_v6.1_backbone.pth',
        'cspdarknet_l': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_l_v6.1_backbone.pth',
        'cspdarknet_x': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_x_v6.1_backbone.pth',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


'''
加载预训练模型
'''


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        print('正在从{}加载预训练模型...'.format(pretrained_model))
        if os.path.exists(pretrained_model):
            para_state_dict = torch.load(pretrained_model)
            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print('{} is not in pretrained model'.format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k].shape):
                    print("[SKIP] shape of pretrained params {} doesn't match.(Pretrained: {}, Actual:{})"
                          .format(k, para_state_dict[k].shape, model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.load_state_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(num_params_loaded, len(model_state_dict),
                                                                     model.__class__.__name__))
        else:
            raise ValueError(
                "The pretrained model directory is not Found: {}".format(pretrained_model)
            )
    else:
        print('No pretrained model to load, {} will be trained from scratch.'.format(model.__class__.__name__))


'''
可视化训练结果，画上gt和prediction的对比图
'''


def draw_boxes(name, img, boxes_det, boxes_gt, save_folder, class_names, num_classes, text_color=(0, 0, 0)):
    '''
    img: np.array格式的图片
    boxes_det: 当前图片所有的检测框，类型是list：[box(class_name, conf, xyxy), box(class_name, conf, xyxy)...]
    boxes_gt: 当前图片所有的真实框，类型是list：[box(class_name, xyxy), box(class_name, xyxy)...]
    save_folder: 保存路径
    ---------------
    return: 无return值，直接保存画好框的图片
    '''
    # 判断保存路径是否存在,没有则新建文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # 拷贝图片
    img_det = img.copy()  # dr(detection_result)
    img_gt = img.copy()  # gt

    # 画框设置不同的颜色
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    box_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    box_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), box_colors))

    # 获取预测框的信息，并画到图片上
    for det in boxes_det:  # 遍历每一行验证集数据
        det = det.split('\n')[0]  # 通过回车分隔
        cls_name = det.split()[0:-5]  # 类别名称
        # 判断一下cls_name类别名称是否以空格为分隔,长度超过1
        if len(cls_name) == 1:
            cls_name = cls_name[0]
        else:
            cls_name = ' '.join(cls_name)
        box = np.zeros(4, dtype=np.uint16)  # 创建长度为4的数组
        box[0] = np.maximum(int(det.split()[-4]), 0)  # x1
        box[1] = np.maximum(int(det.split()[-3]), 0)  # y1
        box[2] = np.minimum(int(det.split()[-2]), img.shape[1] - 1)  # x2
        box[3] = np.minimum(int(det.split()[-1]), img.shape[0] - 1)  # y2
        conf = str(det.split()[-5])[:4]
        # 获取类别的序号(用于给不同的颜色)
        cls_index = class_names.index(cls_name)
        # 将置信度conf的数字加到cls_name(label)后面
        cls_name = cls_name + ' ' + conf

        # 通过cv2进行画矩形框
        thickness = 3
        for i in range(thickness):
            cv2.rectangle(img_det, (box[0] + i, box[1] + i), (box[2] - i, box[3] - i), color=box_colors[cls_index],
                          thickness=1)
        # 绘制标签
        draw_labels(img_det, box, cls_name, label_color=box_colors[cls_index], text_color=text_color)
        # 绘制美化角点
        # draw_box_corner(img_det, box, corner_color)

    # gt部分的绘制(没有置信度信息)
    for det in boxes_gt:
        det = det.split('\n')[0]  # 通过回车分隔
        cls_name = det.split()[0:-4]  # 类别名称(没有置信度信息)
        # 判断一下cls_name类别名称是否以空格为分隔,长度超过1
        if len(cls_name) == 1:
            cls_name = cls_name[0]
        else:
            cls_name = ' '.join(cls_name)
        box = np.zeros(4, dtype=np.uint16)  # 创建长度为4的数组
        box[0] = np.maximum(int(det.split()[-4]), 0)  # x1
        box[1] = np.maximum(int(det.split()[-3]), 0)  # y1
        box[2] = np.maximum(int(det.split()[-2]), 0)  # x2
        box[3] = np.maximum(int(det.split()[-1]), 0)  # y2
        # 获取类别的序号(用于给不同的颜色)
        cls_index = class_names.index(cls_name)
        # 通过cv2进行画矩形框
        thickness = 3
        for i in range(thickness):
            cv2.rectangle(img_gt, (box[0] + i, box[1] + i), (box[2] - i, box[3] - i), color=box_colors[cls_index],
                          thickness=1)
        # 绘制标签
        draw_labels(img_gt, box, cls_name, label_color=box_colors[cls_index], text_color=text_color)
        # 绘制美化角点
        # draw_box_corner(img_gt, box, corner_color)
    # 将两张图拼在一起
    h, w = img.shape[0], img.shape[1]
    img_show = np.zeros((h, w * 2, 3))
    img_show[:, :w, :] = img_det
    img_show[:, w:, :] = img_gt
    img_show = img_show.astype(np.uint16)
    save_path = os.path.join(save_folder, name + '.jpg')
    cv2.imwrite(save_path, img_show)


# 绘制标签
def draw_labels(img, box, label, label_color, text_color):
    """
    :param img: np.array格式的图片
    :param box: [x1,y1,x2,y2]
    :param label: str 标签
    :param label_color: 标签框的背景色，文字默认是黑色
    :param text_color:
    :return: 无返回值，直接写上标签
    """
    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]  # 返回包含文字的框的大小(w,h),baseline
    # 字体默认cv2.FONT_HERSHEY_SIMPLEX
    if box[1] - labelSize[1] - 3 < 0:  # y1-h-3<0(此条件下,在box框内写字)
        cv2.rectangle(img, (box[0], box[1]), (box[0] + labelSize[0], box[1] + labelSize[1] + 3),
                      color=label_color, thickness=-1)  # (x1,y1+2),(x1+w,y1+h+3)用于放字的空间
        cv2.putText(img, label, (box[0], box[1] + labelSize[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=text_color,
                    thickness=1)  # 写上文字
    else:  # 此条件下,在box框外写字
        cv2.rectangle(img, (box[0], box[1] - labelSize[1] - 3), (box[0] + labelSize[0], box[1]),
                      color=label_color, thickness=-1)  # (x1,y1-h-3),(x1+w,y1-3)
        cv2.putText(img, label, (box[0], box[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=text_color, thickness=1)


# 检测框的角点美化
def draw_box_corner(img, box, corner_color):
    """
    :param img: np.array格式的图片
    :param box: [x1 y1 x2 y2]
    :param corner_color: 角点线的颜色
    :return: 无返回值，直接美化框
    """
    box_w = box[2] - box[0]  # x2-x1
    box_h = box[3] - box[1]  # y2-y1
    length_x = (np.round(box_w * 0.1)).astype(np.uint8)  # 角点比例
    length_y = (np.round(box_h * 0.1)).astype(np.uint8)  # 角点比例
    # top left 左上
    cv2.line(img, (box[0], box[1]), (box[0] + length_x, box[1]), corner_color, thickness=1)
    cv2.line(img, (box[0], box[1]), (box[0], box[1] + length_y), corner_color, thickness=1)
    # top right 右上
    cv2.line(img, (box[2], box[1]), (box[2] - length_x, box[1]), corner_color, thickness=1)
    cv2.line(img, (box[2], box[1]), (box[2], box[1] + length_y), corner_color, thickness=1)
    # bottom left 左下
    cv2.line(img, (box[0], box[3]), (box[0] + length_x, box[3]), corner_color, thickness=1)
    cv2.line(img, (box[0], box[3]), (box[0], box[3] - length_y), corner_color, thickness=1)
    # bottom right 右下
    cv2.line(img, (box[2], box[3]), (box[2] - length_x, box[3]), corner_color, thickness=1)
    cv2.line(img, (box[2], box[3]), (box[2], box[3] - length_y), corner_color, thickness=1)


"""
通过多边形得到掩码矩阵
"""


def get_mask(h, w, mask_pt: list):
    # 创建图像
    img = np.zeros((h, w, 1), np.uint8)
    # 遍历每一根多段线
    for pl in mask_pt:
        pl = np.array(pl)
        pl[:, 0] = np.round(pl[:, 0] * w)  # x
        pl[:, 1] = np.round(pl[:, 1] * h)  # y

        # 绘制多边形
        cv2.polylines(img, [np.array(pl, dtype=np.int32)], True, 1)
        # 获取掩码
        img = cv2.fillPoly(img, [np.array(pl, dtype=np.int32)], 1)

    # cv2.imwrite(image.split('.')[0] + '_result.jpg', img)
    return img
