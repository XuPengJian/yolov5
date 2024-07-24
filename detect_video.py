"""
detect_video.py为视频目标追踪的预测功能(可根据具体情况进行修改)
"""
import time
import os
import colorsys

import imageio.v2 as iio
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
# import tensorrt as trt

from nets.yolo import YoloBody
# from utils.trt_tools import TRTModule
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, get_mask, show_config)
from utils.utils_bbox import DecodeBox
from track.tracker.byte_tracker import BYTETracker
from track.trajectory_clustering import draw_lines

import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")  # 忽略版本警告信息

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

'''
推理超参设置
'''


# 现在predict.py中将pth模型转换为onnx模型
# 先将onnx模型-->trt模型
# cd E:\Program Files\TensorRT-8.4.3.1\bin
# trtexec --onnx=E:\gitlab\cars_detection\yolov5\predict_result\onnx_model\20230302model.onnx --saveEngine=E:\gitlab\cars_detection\yolov5\train_result\20230302model.trt --workspace=6000

# mask: [[[0.4022178772966952, 0.0010226039191259938], [0.38190687064935414, 0.08903706537217705], [0.3661094210347555, 0.14545659194464566], [0.3412848573546719, 0.24475495871219044], [0.3221022399655164, 0.3237422959136465], [0.31871707219095957, 0.3891889467377101], [0.3333861325473726, 0.4365813490585838], [0.3469268036456, 0.4862305324423562], [0.3751365351002404, 0.5697314317696097], [0.39996109878032404, 0.655489112159762], [0.41463015913673706, 0.7186789819209269], [0.4191437161694795, 0.7570442599902056], [0.4191437161694795, 0.8089502244368767], [0.4191437161694795, 0.8901943427012315], [0.423657273202222, 0.9466138692737002], [0.4270424409767788, 0.9940062715945737], [0.7034978092322552, 0.9985198337203712], [0.668517742228501, 0.7141654197951294], [0.6752880777776147, 0.655489112159762], [0.6899571381340278, 0.5742449938954073], [0.6989842521995128, 0.5020279998826473], [0.7125249232977402, 0.4185271005553938], [0.7136533125559258, 0.33502620122814025], [0.7023694199740695, 0.25829564508958297], [0.6843151918430997, 0.19284899426551932], [0.6527202926139024, 0.13417268663015194], [0.5985576082209928, 0.003279384982024739]]]
def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    # parser.add_argument('--mode', type=str, default='video',
    #                     help='predict, video, fps, heatmap, export_onnx')
    parser.add_argument('--model_weight', type=str,
                        default=r'train_result/20230306affine.pth',
                        help='模型权重路径')
    parser.add_argument('--trt_weight', type=str,
                        default=r'',  # train_result\20230310model.trt
                        help='trt模型权重路径,如果设置trt路径,则用的是此路径模型')
    parser.add_argument('--phi', type=str, default='s', help='模型使用的版本')

    parser.add_argument('--classes_path', type=str, default='model_data/cars_classes.txt',
                        help='类别对应的txt文件，一般不修改')
    parser.add_argument('--anchors_path', type=str, default='model_data/yolo_anchors.txt',
                        help='先验框对应的txt文件，一般不修改')
    parser.add_argument('--input_shape', type=list, default=[1152, 2048],
                        help='输入网络的分辨率大小[h, w]')  # [1152, 2048]
    parser.add_argument('--letterbox', type=bool, default=False, help='resize图片时是否使用letterbox')
    parser.add_argument('--confidence', type=float, default=0.1,
                        help='检测结果中只有得分大于置信度的预测框会被保留下来')
    parser.add_argument('--nms_iou', type=float, default=0.3, help='非极大抑制所用到的nms_iou大小')
    parser.add_argument('--mask_pt', type=str, default='',
                        help='通过多边形用于框选出用于目标检测的区域,里面是[[[x, y][x, y]][[...]]]的形式')

    parser.add_argument('--video_path', type=str, default=r'D:\gitlab\cars_detection\yolov5\video\DJI_0056_test.mp4',
                        help='视频路径，如果为0的话就是调用摄像头,仅在mode="video"时有效')
    parser.add_argument('--video_fps', type=int, default=30, help='用于保存的视频的fps,仅在mode="video"时有效')
    parser.add_argument('--save_path', type=str, default='predict_result',
                        help='预测的保存文件夹路径，所有检测结果都会保存在这个路径下,此路径下会生成video_output文件夹和txt_output')
    parser.add_argument('--simplify_onnx', type=bool, default=True, help='简化onnx模型，也就是移除常量算子')

    # 目标追踪相关的args
    '''tracking相关的超参'''
    # 跳帧处理
    parser.add_argument("--jump_fps", type=int, default=2, help="跳帧间隔,加快运算速度,最少为1,为默认视频帧数")
    # 两轮匹配结束后，剩下未匹配成功的detections的分数需要大于det_thresh这个阈值才能被初始化为新的track
    parser.add_argument("--track_thresh", type=float, default=0.1, help="tracking confidence threshold")
    # tracks保持为Lost状态（track_buffer）帧就移除
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    # 此处没用到
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    # 检测跟踪结果中的框如果宽高比小于阈值，则视为无效
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=5.0,  # 1.6
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    # 检测跟踪结果中的框如果面积小于min_box_area，则视为无效
    parser.add_argument('--min_box_area', type=float, default=15, help='filter out tiny boxes')  # 10
    # 是否用于检测mot20
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    return parser.parse_args()


"""
检测器(跟踪器使用已包含在内)
"""


class YOLO_PREDICT:
    # 初始化YOLO,通过字典得到变量具体值
    def __init__(self, args, model_path, trt_path, classes_path, anchors_path, input_shape=[640, 640], phi='s',
                 confidence=0.5,
                 nms_iou=0.3, letterbox=True, cuda=True):
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        self.model_path = model_path
        self.trt_path = trt_path
        # self.model_path ='model_data/yolov5_s_v6.1.pth'
        # "model_path": r'E:\gitlab\zoomtoolkit\yolov5\train_result\2022_12_22_11_54_29\Model\best_weights.pth',
        self.classes_path = classes_path
        # anchors_path代表先验框对应的txt文件，一般不修改。
        # anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        self.anchors_path = anchors_path
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # 输入图片的大小，必须为32的倍数。
        self.input_shape = input_shape
        # phi,所使用的YoloV5的版本。n、s、m、l、x
        self.phi = phi
        # 只有得分大于置信度的预测框会被保留下来
        self.confidence = confidence
        # 非极大抑制所用到的nms_iou大小
        self.nms_iou = nms_iou
        # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        # 在多次测试后，发现关闭letterbox_image直接resize的效果更好
        self.letterbox = letterbox
        # 是否使用Cuda
        self.cuda = cuda

        # 获得种类和先验框的数量
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 如果存在pth模型路径切不存在trt路径
        if model_path and not trt_path:
            self.generate()
        # trt模式
        else:
            logger = trt.Logger(trt.Logger.INFO)
            with open(self.trt_path, "rb") as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())  # 输入trt本地文件，返回ICudaEngine对象
            # for idx in range(engine.num_bindings):  # 查看输入输出的名字，类型，大小
            #     is_input = engine.binding_is_input(idx)
            #     name = engine.get_binding_name(idx)
            #     op_type = engine.get_binding_dtype(idx)
            #     shape = engine.get_binding_shape(idx)
            #     print('input id:', idx, ' is input: ', is_input, ' binding name:', name, ' shape:', shape, 'type: ', op_type)

            # 这里填的是input和output的binding name
            self.trt_model = TRTModule(engine, ["images"], ["610", "611", "output"])

        # show_config(**self._defaults)

        # 用于生成检测的mask掩码矩阵,相对于输入网络的input_shape来生成,如果mask_pt这个参数存在的话,否则为全图
        if args.mask_pt:
            self.mask = get_mask(input_shape[0], input_shape[1], eval(args.mask_pt))
        else:
            self.mask = np.array([])

        # 视频帧率
        self.video_fps = args.video_fps
        # 实例化跟踪器
        self.tracker = BYTETracker(args, frame_rate=self.video_fps)
        self.frame_id = 0

    # 生成模型
    def generate(self, onnx=False):
        # 建立yolo模型，载入yolo模型的权重
        self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # 检测每一帧图片,并进行预测得到车辆id
    def detect_image(self, image, txt_save_path):
        """
        :param image: 用于目标检测的图片PIL格式
        :param txt_save_path: 保存检测结果txt的路径
        :return: image: 检测并完成目标跟踪的image
        """
        # image为PIL格式
        # image_data为cv2格式转为tensor格式

        # 计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        # 是否是用letterbox_image
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox)
        # 添加上batch_size维度,并将HWC->CHW
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)  # 转化为torch格式
            if self.cuda:
                images = images.cuda()
            # 将图像输入网络当中进行预测！
            outputs = self.net(images)
            # outputs是tuple,包含三个特征层,每个特征层是tensor格式,shape为[batch_size,(5+cls)*3,grid_h,grid_w],依次是大目标(大网格数量少),中目标,小目标(小网格数量多)

            outputs = self.bbox_util.decode_box(outputs)
            # 将预测框进行堆叠，然后进行非极大抑制(并还原到原图大小分辨率)
            results = self.bbox_util.non_max_suppression_all(torch.cat(outputs, 1), self.num_classes,
                                                             self.input_shape,
                                                             image_shape, self.letterbox, conf_thres=self.confidence,
                                                             nms_thres=self.nms_iou)
            # 返回的是单张图片预测出来的框的信息组合成的列表，数据格式为列表List[None] List[np.shape(num_anchors, 6)],列表的长度为batch_szie的大小
            # [y1 x1 y2 x2 obj_cls_conf(物体*类别置信度) class_pred(类别序号)]

            # 这个做法可以选取对应的类型用于目标跟踪!
            # det = torch.index_select(det[0].clone().cpu(), dim=0,
            #                          index=(torch.where(det[0][:, 5:6] == 4)[0]).cpu())

            # 取出列表第一个batch的数据(不考虑分类)
            det = results[0]  # numpy格式,shape(num_anchors, 6)
            # 内部为[y1 x1 y2 x2 obj_cls_conf(物体*类别置信度) class_pred(类别序号)]

            # 如果存在det数据(det为numpy数据), 则进行mask处理
            if det is not None:
                # 在此处通过mask过滤掉中心点不在mask框内的检测结果,mask是numpy格式
                if len(self.mask):
                    # 计算中心点(这里要除一下输入图像与网络输出的缩放倍数)
                    # 这里的mask是基于input_shape来做的, 而center_x和center_y是已经还原到原图大小
                    center_x = (det[:, 1] + det[:, 3]) / 2 / image_shape[1] * self.input_shape[1]
                    center_y = (det[:, 0] + det[:, 2]) / 2 / image_shape[0] * self.input_shape[0]
                    # 在mask中找到中心点相应位置的元素(先y,再x) --> 得到给anchors格式的掩码
                    det_mask = self.mask[center_y.astype('int32'), center_x.astype('int32')]
                    # 过滤信息
                    det = det[(det_mask == 1).squeeze()]

            # 如果通过mask过滤之后仍然存在det数据(det为numpy数据)
            if det is not None:
                # [y1,x1,y2,x2]->[x1,y1,x2,y2]两两交换位置
                det[:, 0], det[:, 1] = det[:, 1], det[:, 0].copy()  # 这里的copy用于防止直接链到原表,这样才能交换成功
                det[:, 2], det[:, 3] = det[:, 3], det[:, 2].copy()  # 这里的copy用于防止直接链到原表,这样才能交换成功
                if det[:, :4] is not None:
                    # 更新跟踪器, 需要输入[h,w]
                    online_targets = self.tracker.update(det, [self.input_shape[0], self.input_shape[1]], (800, 1440))
                    online_tlwhs = []  # 有效追踪的 尺寸x1 y1 width height
                    online_ids = []  # 有效追踪的 id
                    online_scores = []  # 有效追踪的 物体置信度*类别置信度
                    online_cls = []  # 有效追踪的 类别
                    # 获取每一个track
                    for t in online_targets:
                        # 获取track中框的尺寸
                        tlwh = t.tlwh
                        # 获取track的id
                        tid = t.track_id
                        # 如果宽高比大于阈值，则vertical为True
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        # 只有当框的面积大于min_box_area，且框的宽高比小于aspect_ratio_thresh，才被视为有效
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_cls.append(t.cls)

                    # -----------------------------------
                    # ----------用自己的方法绘制图----------
                    # -----------------------------------
                    # 设置字体与边框厚度
                    font = ImageFont.truetype(font='model_data/simhei.ttf',
                                              size=np.floor(1e-2 * image_shape[0] + 0.5).astype('int32'))
                    # fps_font = ImageFont.truetype(font='model_data/simhei.ttf',
                    #                               size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                    thickness = int(max((image_shape[1] + image_shape[0]) // np.mean(self.input_shape), 1))  # 最小不能小过1

                    # 遍历每帧所有有效激活类别
                    for i, c in list(enumerate(online_cls)):
                        # 基础框信息格式化
                        predicted_class = self.class_names[int(c)]  # 得到类别名称
                        tlwh = online_tlwhs[i]  # 得到对应的框信息 x1 y1 w h
                        score = online_scores[i]  # 得到对应的置信度分数

                        # 格式化框数据
                        x1, y1, w, h = tlwh
                        left, top, right, bottom = x1, y1, x1 + w, y1 + h  # x1 y1 x2 y2

                        # 四舍五入转化为整数,并防止超出切割图片范围
                        top = max(0, np.floor(top).astype('int32'))  # y1
                        left = max(0, np.floor(left).astype('int32'))  # x1
                        bottom = min(image_shape[0], np.floor(bottom).astype('int32'))  # y2
                        right = min(image_shape[1], np.floor(right).astype('int32'))  # x2

                        # 赋予id
                        obj_id = int(online_ids[i])
                        id_text = '{}'.format(int(obj_id))

                        # 格式化类别名称和分数
                        # label = '{} {:.2f}'.format(predicted_class, score)
                        label = '{}[{}]'.format(predicted_class, obj_id)
                        draw = ImageDraw.Draw(image)  # PIL绘制图案,直接以image为背景进行绘画
                        label_size = draw.textsize(label, font)  # 字体大小
                        label = label.encode('utf-8')  # 字符编码

                        # -----------------------------------
                        # ----------将结果写入txt文件中--------
                        # -----------------------------------
                        # 数据格式为,每行数据: fps,id,x1,y1,x2,y2,conf,cls
                        with open(txt_save_path, 'a') as f:
                            f.write(
                                f'{self.frame_id + 1},{obj_id},{left},{top},{right},{bottom},{score:.6f},{int(c)}\n')

                        # 绘制图像
                        # 确定写字的原点位置
                        if top - label_size[1] >= 0:  # y1 - h>=0
                            text_origin = np.array([left, top - label_size[1]])  # x1 y1-h  # 写在下面
                        else:
                            text_origin = np.array([left, top + 1])  # x1 y1+1 写在上面

                        # 遍历不同的粗细(通过重复画几次不同的偏移量来表现粗细)
                        for i in range(thickness):
                            # 绘制矩形,向内偏移线的粗细
                            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[int(c)])
                        # 绘制用于放文字的框,填充背景
                        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(c)])
                        # 绘制文字label
                        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

                        # 获取检测的fps文字，并显示在图片上
                        # fps = (fps + (1. / (time.time() - t1))) / 2
                        # fps = 'frame: %d fps: %.2f num: %d' % (self.frame_id + 1, self.video_fps, len(online_tlwhs))
                        # draw.text((20, 20), fps, fill=(250, 30, 30), font=fps_font)

                        # 删除画板
                        del draw

            # 返回结果,如果结果为空,就直接返回原图
            return image

    def detect_image_trt(self, image, txt_save_path):
        """
        :param image: 用于目标检测的图片PIL格式
        :param txt_save_path: 保存检测结果txt的路径
        :return: image: 检测并完成目标跟踪的image
        """
        # image为PIL格式
        # image_data为cv2格式转为tensor格式

        # 计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        # 是否是用letterbox_image
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox)
        # 添加上batch_size维度,并将HWC->CHW(np格式)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # 转化为torch格式
        img_input = torch.from_numpy(image_data)
        img_input = img_input.unsqueeze(0)
        img_input = img_input.to('cuda')
        # 运行模型
        result_trt = self.trt_model(img_input)
        out2, out1, out0 = result_trt
        out2 = out2[0]
        out1 = out1[0]
        out0 = out0[0]
        outputs = (out0, out1, out2)
        # outputs是tuple,包含三个特征层,每个特征层是tensor格式,shape为[batch_size,(5+cls)*3,grid_h,grid_w],依次是大目标(大网格数量少),中目标,小目标(小网格数量多)

        outputs = self.bbox_util.decode_box(outputs)
        # 将预测框进行堆叠，然后进行非极大抑制(并还原到原图大小分辨率)
        results = self.bbox_util.non_max_suppression_all(torch.cat(outputs, 1), self.num_classes,
                                                         self.input_shape,
                                                         image_shape, self.letterbox, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
        # 返回的是单张图片预测出来的框的信息组合成的列表，数据格式为列表List[None] List[np.shape(num_anchors, 6)],列表的长度为batch_szie的大小
        # [y1 x1 y2 x2 obj_cls_conf(物体*类别置信度) class_pred(类别序号)]

        # 这个做法可以选取对应的类型用于目标跟踪!
        # det = torch.index_select(det[0].clone().cpu(), dim=0,
        #                          index=(torch.where(det[0][:, 5:6] == 4)[0]).cpu())

        # 取出列表第一个batch的数据(不考虑分类)
        det = results[0]  # numpy格式,shape(num_anchors, 6)
        # 内部为[y1 x1 y2 x2 obj_cls_conf(物体*类别置信度) class_pred(类别序号)]

        # 如果存在det数据(det为numpy数据), 则进行mask处理
        if det is not None:
            # 在此处通过mask过滤掉中心点不在mask框内的检测结果,mask是numpy格式
            if len(self.mask):
                # 计算中心点(这里要除一下输入图像与网络输出的缩放倍数)
                # 这里的mask是基于input_shape来做的, 而center_x和center_y是已经还原到原图大小
                center_x = (det[:, 1] + det[:, 3]) / 2 / image_shape[1] * self.input_shape[1]
                center_y = (det[:, 0] + det[:, 2]) / 2 / image_shape[0] * self.input_shape[0]
                # 在mask中找到中心点相应位置的元素(先y,再x) --> 得到给anchors格式的掩码
                det_mask = self.mask[center_y.astype('int32'), center_x.astype('int32')]
                # 过滤信息
                det = det[(det_mask == 1).squeeze()]

        # 如果通过mask过滤之后仍然存在det数据(det为numpy数据)
        if det is not None:
            # [y1,x1,y2,x2]->[x1,y1,x2,y2]两两交换位置
            det[:, 0], det[:, 1] = det[:, 1], det[:, 0].copy()  # 这里的copy用于防止直接链到原表,这样才能交换成功
            det[:, 2], det[:, 3] = det[:, 3], det[:, 2].copy()  # 这里的copy用于防止直接链到原表,这样才能交换成功
            if det[:, :4] is not None:
                # 更新跟踪器, 需要输入[h,w]
                online_targets = self.tracker.update(det, [self.input_shape[0], self.input_shape[1]], (800, 1440))
                online_tlwhs = []  # 有效追踪的 尺寸x1 y1 width height
                online_ids = []  # 有效追踪的 id
                online_scores = []  # 有效追踪的 物体置信度*类别置信度
                online_cls = []  # 有效追踪的 类别
                # 获取每一个track
                for t in online_targets:
                    # 获取track中框的尺寸
                    tlwh = t.tlwh
                    # 获取track的id
                    tid = t.track_id
                    # 如果宽高比大于阈值，则vertical为True
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    # 只有当框的面积大于min_box_area，且框的宽高比小于aspect_ratio_thresh，才被视为有效
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_cls.append(t.cls)

                # -----------------------------------
                # ----------用自己的方法绘制图----------
                # -----------------------------------
                # 设置字体与边框厚度
                font = ImageFont.truetype(font='model_data/simhei.ttf',
                                          size=np.floor(1e-2 * image_shape[0] + 0.5).astype('int32'))
                # fps_font = ImageFont.truetype(font='model_data/simhei.ttf',
                #                               size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = int(max((image_shape[1] + image_shape[0]) // np.mean(self.input_shape), 1))  # 最小不能小过1

                # 遍历每帧所有有效激活类别
                for i, c in list(enumerate(online_cls)):
                    # 基础框信息格式化
                    predicted_class = self.class_names[int(c)]  # 得到类别名称
                    tlwh = online_tlwhs[i]  # 得到对应的框信息 x1 y1 w h
                    score = online_scores[i]  # 得到对应的置信度分数

                    # 格式化框数据
                    x1, y1, w, h = tlwh
                    left, top, right, bottom = x1, y1, x1 + w, y1 + h  # x1 y1 x2 y2

                    # 四舍五入转化为整数,并防止超出切割图片范围
                    top = max(0, np.floor(top).astype('int32'))  # y1
                    left = max(0, np.floor(left).astype('int32'))  # x1
                    bottom = min(image_shape[0], np.floor(bottom).astype('int32'))  # y2
                    right = min(image_shape[1], np.floor(right).astype('int32'))  # x2

                    # 赋予id
                    obj_id = int(online_ids[i])
                    id_text = '{}'.format(int(obj_id))

                    # 格式化类别名称和分数
                    # label = '{} {:.2f}'.format(predicted_class, score)
                    label = '{}[{}]'.format(predicted_class, obj_id)
                    draw = ImageDraw.Draw(image)  # PIL绘制图案,直接以image为背景进行绘画
                    label_size = draw.textsize(label, font)  # 字体大小
                    label = label.encode('utf-8')  # 字符编码

                    # -----------------------------------
                    # ----------将结果写入txt文件中--------
                    # -----------------------------------
                    # 数据格式为,每行数据: fps,id,x1,y1,x2,y2,conf,cls
                    with open(txt_save_path, 'a') as f:
                        f.write(
                            f'{self.frame_id + 1},{obj_id},{left},{top},{right},{bottom},{score:.6f},{int(c)}\n')

                    # 绘制图像
                    # 确定写字的原点位置
                    if top - label_size[1] >= 0:  # y1 - h>=0
                        text_origin = np.array([left, top - label_size[1]])  # x1 y1-h  # 写在下面
                    else:
                        text_origin = np.array([left, top + 1])  # x1 y1+1 写在上面

                    # 遍历不同的粗细(通过重复画几次不同的偏移量来表现粗细)
                    for i in range(thickness):
                        # 绘制矩形,向内偏移线的粗细
                        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[int(c)])
                    # 绘制用于放文字的框,填充背景
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(c)])
                    # 绘制文字label
                    draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

                    # 获取检测的fps文字，并显示在图片上
                    # fps = (fps + (1. / (time.time() - t1))) / 2
                    # fps = 'frame: %d fps: %.2f num: %d' % (self.frame_id + 1, self.video_fps, len(online_tlwhs))
                    # draw.text((20, 20), fps, fill=(250, 30, 30), font=fps_font)

                    # 删除画板
                    del draw

        # 返回结果,如果结果为空,就直接返回原图
        return image


"""
主程序
"""


def main(args):
    # 实例化检测器(跟踪器实例化也写入内部)
    yolo = YOLO_PREDICT(args, model_path=args.model_weight, trt_path=args.trt_weight, classes_path=args.classes_path,
                        anchors_path=args.anchors_path, input_shape=args.input_shape, phi=args.phi,
                        confidence=args.confidence, nms_iou=args.nms_iou, letterbox=args.letterbox)

    start = time.time()
    # print('------开始计算------')
    capture = cv2.VideoCapture(args.video_path)  # 加载视频
    video_save_folder = os.path.join(args.save_path, 'video_output')  # 视频保存路径
    txt_save_folder = os.path.join(args.save_path, 'txt_output')  # 视频结果txt保存路径
    # 如果路径不存在则新建文件夹
    if not os.path.exists(video_save_folder):
        os.makedirs(video_save_folder)
    if not os.path.exists(txt_save_folder):
        os.makedirs(txt_save_folder)
    # 基于basename,组合出txt的名字
    txt_save_path = os.path.join(txt_save_folder, os.path.basename(args.video_path).split('.')[0] + '.txt')
    file = open(txt_save_path, 'w')
    file.close()

    # 基于basename,组合出video的名字(强制为mp4后缀)
    video_save_path = os.path.join(video_save_folder, os.path.basename(args.video_path).split('.')[0] + '.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'h264')  # MPEG-4编码类型的编码器,用于决定写入的视频的编码类型
    out = iio.get_writer(video_save_path, format="ffmpeg", mode='I', fps=25, codec='libx264', pixelformat='yuv420p')
    # 获取视频的宽高
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 这里决定视频的保存路径,编码类型,帧率,以及画面尺寸
    # out = cv2.VideoWriter(video_save_path, fourcc, args.video_fps, size)

    # read()方法按帧读取视频，返回ret, frame，ret表示是否正确读取帧，frame就是帧图片
    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    tmp_frame = None  # 记录上一帧

    # 遍历视频帧
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()

        # 如果当ref的状态不为True(也就是要结束读取视频，读到最后一张了)
        if not ref:
            # 选取最后一帧以及最终生成的txt文件生成车流量的图片
            # 已经是rgb了甚至都不有再转一下了
            # tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2RGB)
            iio.imwrite(txt_save_path.split('.')[0] + '.jpg', tmp_frame)
            # 尝试生成轨迹
            try:
                count_result, front_colors = draw_lines(tmp_frame, txt_save_path, threshold=0.125, min_cars=5)
                # print(count_result, front_colors)
                # with open(txt_save_path.split('.')[0] + '_result.txt', 'w') as f:
                #     f.write(f'{count_result}\n{front_colors}\n')
                # print("Camera capture over!")
                break
            # 如果轨迹无法生成
            except Exception as e:
                with open(txt_save_path.split('.')[0] + '_result.txt', 'w') as f:
                    f.write(f'0\n0\n')
                print(e)
                print('轨迹图像生成失败!')
                break

        # 读取成功则开始计算frame_id
        yolo.frame_id += 1
        # 跳帧进行一次运算
        if not int(yolo.frame_id) % args.jump_fps == 0:
            continue

        # 格式转变，BGRtoRGB,image转为PIL格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        # ---------------------------------
        # -----------进行检测!--------------
        # ----------------------------------
        # 如果trt路径存在
        if args.trt_weight:
            frame = np.array(yolo.detect_image_trt(frame, txt_save_path))
        # 如果trt路径不存在则用普通pth模型格式进行检测
        else:
            frame = np.array(yolo.detect_image(frame, txt_save_path))

        # RGBtoBGR满足opencv显示格式
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        tmp_frame = frame  # 用于存最新的一帧,这样在下一帧才能使用
        # 这里是获取检测的fps，并显示在图片上
        # fps = (fps + (1. / (time.time() - t1))) / 2
        # print("fps= %.2f" % (fps))
        # frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 展示图片(注释掉则不需要展示)
        # cv2.imshow("video", frame)
        # 展示的图片等待1ms
        # c = cv2.waitKey(1) & 0xff

        if video_save_path != "":
            out.append_data(frame)
            # out.write(frame)

        # Esc的ASCII码为27，表示如果按下esc，则停止检测
        # if c == 27:
        #     capture.release()
        #     break

    # print("Video Detection Done!")
    capture.release()
    # print("Save processed video to the path :" + video_save_path)
    # out.release()
    out.close()
    cv2.destroyAllWindows()
    end = time.time()
    # print('------计算完成------')
    # print(f'计算总时长为: {end - start}s')


if __name__ == "__main__":
    args = parse_args()
    main(args)
