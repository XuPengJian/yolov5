import datetime
import os

import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir  # 用于保存loss相关文件的路径

        self.train_loss = []  # 用于储存训练集的loss
        self.train_box_loss = []  # 训练集box_loss
        self.train_cls_loss = []  # 训练集cls_loss
        self.train_conf_loss = []  # 训练集conf_loss

        self.val_loss = []  # 用于储存验证集的loss
        self.val_box_loss = []  # 验证集box_loss
        self.val_cls_loss = []  # 验证集cls_loss
        self.val_conf_loss = []  # 验证集conf_loss

        os.makedirs(self.log_dir)  # 创建用于存放loss的文件夹
        self.writer = SummaryWriter(self.log_dir)  # tensorboard的绘制
        # 绘制网络的结构图
        try:
            # batch_size, channel, input_shape[0], input_shape[1]
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, train_loss, val_loss, avg_losses, val_avg_losses):
        # 创建用于存放loss的文件夹
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 拿到loss的具体项
        train_box_loss = avg_losses[0].item()
        train_cls_loss = avg_losses[1].item()
        train_conf_loss = avg_losses[2].item()
        # 验证集
        val_box_loss = val_avg_losses[0].item()
        val_cls_loss = val_avg_losses[1].item()
        val_conf_loss = val_avg_losses[2].item()

        self.train_loss.append(train_loss)  # 将训练集的loss存在列表中
        self.val_loss.append(val_loss)  # 将验证集的loss存在列表中

        self.train_box_loss.append(train_box_loss)
        self.train_cls_loss.append(train_cls_loss)
        self.train_conf_loss.append(train_conf_loss)

        self.val_box_loss.append(val_box_loss)
        self.val_cls_loss.append(val_cls_loss)
        self.val_conf_loss.append(val_conf_loss)

        # 将loss结果写入到txt中
        with open(os.path.join(self.log_dir, "epoch_train_loss.txt"), 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        with open(os.path.join(self.log_dir, "train_box_loss.txt"), 'a') as f:
            f.write(str(train_box_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "train_cls_loss.txt"), 'a') as f:
            f.write(str(train_cls_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "train_conf_loss.txt"), 'a') as f:
            f.write(str(train_conf_loss))
            f.write("\n")

        with open(os.path.join(self.log_dir, "val_box_loss.txt"), 'a') as f:
            f.write(str(val_box_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "val_cls_loss.txt"), 'a') as f:
            f.write(str(val_cls_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "val_conf_loss.txt"), 'a') as f:
            f.write(str(val_conf_loss))
            f.write("\n")

        # 添加到tensorboard展示面板
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)

        self.writer.add_scalar('train_box_loss', train_box_loss, epoch)
        self.writer.add_scalar('val_box_loss', val_box_loss, epoch)

        self.writer.add_scalar('train_cls_loss', train_cls_loss, epoch)
        self.writer.add_scalar('val_cls_loss', val_cls_loss, epoch)

        self.writer.add_scalar('train_conf_loss', train_conf_loss, epoch)
        self.writer.add_scalar('val_conf_loss', val_conf_loss, epoch)

        # 绘制loss的曲线
        self.loss_plot()
        self.each_loss_plot()

    # 绘制train和val的loss曲线
    def loss_plot(self):
        iters = range(1, len(self.train_loss) + 1)

        # 创建画布
        plt.figure()
        # 绘制loss和val_loss
        plt.plot(iters, self.train_loss, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.train_loss) < 25:
                num = 5
            else:
                num = 15
            # 绘制平滑后的loss和val_loss
            plt.plot(iters, scipy.signal.savgol_filter(self.train_loss, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        # 绘制其他的一些细节
        plt.grid(True)  # 是否带背景网格
        plt.xlabel('Epoch')  # x轴变量名称
        plt.ylabel('Loss')  # y轴变量名称
        plt.legend(loc="upper right")  # 在右上角绘制图例标签

        # 保存图片到所在路径
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变
        plt.close("all")  # 关闭 window，如果没有指定，则指当前 window。

    # 绘制train和val中的box,cls,conf曲线
    def each_loss_plot(self):
        iters = range(1, len(self.train_box_loss) + 1)
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8), sharex=True)
        # train_box_loss
        ax[0, 0].plot(iters, self.train_box_loss)
        ax[0, 0].set_title('train_box_loss')
        # train_cls_loss
        ax[0, 1].plot(iters, self.train_cls_loss)
        ax[0, 1].set_title('train_cls_loss')
        # train_conf_loss
        ax[0, 2].plot(iters, self.train_conf_loss)
        ax[0, 2].set_title('train_conf_loss')

        # val_box_loss
        ax[1, 0].plot(iters, self.val_box_loss)
        ax[1, 0].set_title('val_box_loss')
        # val_cls_loss
        ax[1, 1].plot(iters, self.val_cls_loss)
        ax[1, 1].set_title('val_cls_loss')
        # val_cls_loss
        ax[1, 2].plot(iters, self.val_conf_loss)
        ax[1, 2].set_title('val_conf_loss')
        # 保存图片到所在路径
        plt.savefig(os.path.join(self.log_dir, "each_loss.png"))

        plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变
        plt.close("all")  # 关闭 window，如果没有指定，则指当前 window。


class EvalCallback():
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, cuda, \
                 map_out_path="map_out_path", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True,
                 MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net  # 网络模型
        self.input_shape = input_shape  # 输入网络的shape 比如[640,640]
        self.anchors = anchors  # 候选框
        self.anchors_mask = anchors_mask  # 候选框筛选方式
        self.class_names = class_names  # 类别名称(列表)
        self.num_classes = num_classes  # 类别数量
        self.val_lines = val_lines  # 验证集信息(图片路径 x1 y1 x2 y2 cls)
        self.log_dir = log_dir  # map结果保存路径
        self.cuda = cuda  # 使用的设备
        self.map_out_path = os.path.join(self.log_dir, map_out_path)  # map生成临时文件保存路径,放在对应的Log底下
        self.max_boxes = max_boxes  # 最大框数量
        self.confidence = confidence  # 置信度阈值
        self.nms_iou = nms_iou  # 非极大值抑制
        self.letterbox_image = letterbox_image  # 是否通过letter_box的方式来缩放图片到指定尺寸(会有灰色的边),否则就是直接使用resize的方法
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag  # 是否在训练时进行验证（验证集），安装pycocotools库后，评估体验更佳
        self.period = period  # map获取周期

        # 实例化box数据解码器
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        self.maps = [0]  # 用于储存计算好的map值(起点数据为0)
        self.epoches = [0]  # 用于储存对于的epoch数据(起点数据为0)
        # 采用安装pycocotools库模式
        if self.eval_flag:
            # 创建记录map值的txt,并先把初始值0记录在内
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    # 获得预测txt(直接打印出来了)
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """
        :param image_id: 图片名称/图片id
        :param image: 图片数据PIL格式shape为(H,W,C)
        :param class_names: 类别名称(列表按顺序)
        :param map_out_path: map_out_path保存文件路径
        :return:
        """
        # 打开detection-results文件夹,并将结果写入对于的txt文件
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])  # 获取图片的[H W]
        # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        # 给图像增加灰条，实现不失真的resize(使用letter_box的方式resize图片)
        # 如果letterbox_image的参数为False,则直接使用resize进行识别
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 将图片img归一化在0到1之间,并从[H W C]->[C H W],并将数据都转变为numpy的np.float32,并添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # 在不计算梯度的环境下
        with torch.no_grad():
            # 将numpy
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # 将图像输入网络当中进行预测！
            outputs = self.net(images)
            # 对输出结果进行解码,输出结果为三个特征图的列表,框数据(x y w h)归一化到0-1
            # [(batch_size, num_anchors1, 85),(batch_size, num_anchors2, 85),(batch_size, num_anchors3, 85)]
            outputs = self.bbox_util.decode_box(outputs)
            # 将预测框进行堆叠(候选框,特征图格点数量信息统合,融合成框的数量)，然后进行非极大抑制
            # 返回的是单张图片预测出来的框的信息组合成的列表，数据格式为列表[None] [[num_anchors, 7][num_anchors, 7]],列表的长度为batch_szie的大小
            # [y1 x1 y2 x2 obj_conf(物体置信度)  class_conf(类别置信度)  class_pred(类别序号)]
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            # 如果返回的实际结果是None,直接跳过,返回return
            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')  # 获取class_pred(类别序号),位于最后一位
            top_conf = results[0][:, 4] * results[0][:, 5]  # obj_conf * class_conf
            top_boxes = results[0][:, :4]  # 获取框数据y1 x1 y2 x2

        # 将置信度从大到小排序,并取出前max_boxes数量(默认是100)的框,获取top_100序号列表
        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]  # 取出top_100的框y1 x1 y2 x2
        top_conf = top_conf[top_100]  # 取出top_100的置信度conf
        top_label = top_label[top_100]  # 取出top_100的类别序号class_pred

        # 遍历top_100
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]  # 基于类别序号获取框的类别名称
            box = top_boxes[i]  # 获取对应框数据y1 x1 y2 x2
            score = str(top_conf[i])  # 获取对应的置信度分数(转化为字符型)

            top, left, bottom, right = box  # y1 x1 y2 x2
            # 如果预测出来的类别不在class_names中,则直接跳过
            if predicted_class not in class_names:
                continue

            # 写入结果在txt中(预测类别名称, 置信度分数(物体置信度*类别置信度,保留6位有效数字), x1, y1, x2, y2)
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        # 关闭文件夹
        f.close()
        return

    def on_epoch_end(self, epoch, model_eval):
        # 如果epoch数,满足记录周期,并且启用eval_flag模式
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval  # 网络使用eval模式下,不计算梯度的模型

            # 检查文件夹map_out_path是否存在,并新建相应的文件夹
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")

            # 这里也会显示mAP计算过程的进度条(每张验证集的图片)
            for annotation_line in tqdm(self.val_lines):
                # 逐行读取数据
                line = annotation_line.split()  # 基于空格or回车进行划分
                image_id = os.path.basename(line[0]).split('.')[0]  # 获取image的名字
                # 读取图像并转换成RGB图像(PIL格式,通过路径读取)
                image = Image.open(line[0]).convert('RGB')
                # 获得gt框[[x1 y1 x2 y2 cls] [...]]
                gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                # 获得预测txt(直接写进txt了)
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)

                # 获得真实框txt
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]  # 获取预测类别名称
                        # 写入结果在txt中(预测类别名称, 置信度分数(物体置信度*类别置信度,保留6位有效数字), x1, y1, x2, y2)
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            # 计算Map得分
            print("Calculate Map.")
            try:
                # 利用pycocotools工具计算map值
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            self.maps.append(temp_map)  # 将计算出的map加入到列表中
            self.epoches.append(epoch)  # 将当前epoch也加入到列表中

            # 将Map计算结果写入到txt当中
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            # 绘制Map图像
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            # shutil.rmtree(self.map_out_path)
