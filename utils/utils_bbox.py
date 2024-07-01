import numpy as np
import torch
from torchvision.ops import nms


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors  # 9个候选框
        self.num_classes = num_classes  # 类别数量
        self.bbox_attrs = 5 + num_classes  # 框属性数量
        self.input_shape = input_shape  # 输入网络的图片大小[60,60]
        self.anchors_mask = anchors_mask  # 候选框的筛选掩码[[大3][中3][小3]]
        # -----------------------------------------------------------#
        #   筛选结果如下:
        #   20x20的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   40x40的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   80x80的特征层对应的anchor是[10,13],[16,30],[33,23]
        # -----------------------------------------------------------#

    """
    对预测结果的框进行解码,解码得到的框(x y w h)数据归一化到0-1之间,并且三个特征图形成一个列表,(格子数*候选框数)已经合并
    """

    def decode_box(self, inputs):
        outputs = []  # 新建一个储存最终结果(包含三个特征图的结果)的列表
        # 遍历三个特征图
        for i, input in enumerate(inputs):
            # -----------------------------------------------#
            #   输入的input一共有三个(对应三个特征图)，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (4 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            # -----------------------------------------------#
            batch_size = input.size(0)  # 获取bs
            input_height = input.size(2)  # 获取特征图的h
            input_width = input.size(3)  # 获取特征图的w
            # 其中channel是input.size(1),记录的是网络预测结果的维度
            # -----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = self.input_shape[0] / input_height  # h与原图倍数关系
            stride_w = self.input_shape[1] / input_width  # w与原图倍数关系

            # 此时获得的scaled_anchors大小是相对于特征层的(将候选框缩放到特征图的尺寸)
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]

            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            # 将85的数据从255(85*3)里拆开,并放在最后一个维度
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # 通过torch.sigmoid实现归一化
            # 先验框的中心位置的调整参数
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            # 先验框的宽高调整参数
            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])
            # 获得置信度，是否有物体
            conf = torch.sigmoid(prediction[..., 4])
            # 种类置信度
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 生成网格，先验框中心，网格左上角
            # batch_size,3,20,20
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # 按照网格格式生成先验框的宽高
            # batch_size,3,20,20
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))  # 先验框的w
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))  # 先验框的h
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整,重新映射
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # ----------------------------------------------------------#
            # 创建一个浮点型tensor格式,x y w h
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            # 重新映射,并还原到特征图的尺寸上
            pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

            # 将输出结果归一化成小数的形式
            # 归一化比例
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            # 归一化并将置信度conf,预测分类cls沿着最后一个维度拼接起来(batch_size, 3, 20, 20, 85)->(batch_size, num_anchors, 85)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)  # 将此特征图预测的结果添加到outputs中
        return outputs  # 输出三个特征图的结果(列表的格式)

    """
    得到对应原图的框
    """

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        """
        :param box_xy: 框的xy数据(格式为0-1),numpy格式
        :param box_wh: 框的wh数据(格式为0-1),numpy格式
        :param input_shape: 输入网络的shape [640 640]
        :param image_shape: 图片的实际高宽 [H W]
        :param letterbox_image: 是否通过letter_box的方式来缩放图片到指定尺寸(会有灰色的边),否则就是直接使用resize的方法
        :return:boxes: [y1,x1,y2,x2]
        """
        # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
        box_yx = box_xy[..., ::-1]  # yx
        box_hw = box_wh[..., ::-1]  # hw
        input_shape = np.array(input_shape)  # 输入网络的shape [640 640]
        image_shape = np.array(image_shape)  # 图片的实际高宽[H W]

        if letterbox_image:
            # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            # new_shape指的是宽高缩放情况(无失真的缩放到满足小于或等于input_shape的情况)
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))  # 用偏小的比例作为缩放依据
            offset = (input_shape - new_shape) / 2. / input_shape  # 偏移量并且做归一化后的结果
            scale = input_shape / new_shape  # 缩放量(映射到640得到shape的scale值,目的是弄成没有letterbox的数据格式)

            box_yx = (box_yx - offset) * scale  # 还原框的偏移数据(相当于往左上角移动,并将yx缩放成input_shape(640)的尺寸)
            box_hw *= scale  # 还原框大小的缩放量(hw缩放成input_shape(640)的尺寸)

        box_mins = box_yx - (box_hw / 2.)  # 左上角(归一化结果)y1 x1
        box_maxes = box_yx + (box_hw / 2.)  # 右下角(归一化结果)y2 x2
        # 将计算结果得到的y1 x1 y2 x2仍然限制在0-1之间
        box_mins = np.maximum(box_mins, 0)
        box_maxes = np.minimum(box_maxes, 1)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)  # [y1,x1,y2,x2]
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)  # 将归一化的数据分别乘上原图image_shape[h w h w]
        return boxes

    """
    非极大值抑制
    """

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        """
        :param prediction:预测结果outputs沿着维度1拼接(batch_size, num_anchors, 85),框(x y w h)数据已经归一化
        :param num_classes:类别数量
        :param input_shape:输入网络的shape [640 640]
        :param image_shape:图片的实际高宽 [H W]
        :param letterbox_image:是否通过letter_box的方式来缩放图片到指定尺寸(会有灰色的边),否则就是直接使用resize的方法
        :param conf_thres:置信度阈值
        :param nms_thres:非极大值抑制
        :return: 返回的是单张图片预测出来的框的信息组合成的列表，数据格式为列表[None] [[num_anchors, 7][num_anchors, 7]],列表的长度为batch_szie的大小
        [y1 x1 y2 x2 obj_conf(物体置信度)  class_conf(类别置信度)  class_pred(类别序号)]
        """
        # 将预测结果的格式转换成左上角右下角的格式。[x,y,w,h]-->[x1,y1,x2,y2]
        # prediction  [batch_size, num_anchors, 85]
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1 = x - w/2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1 = y - h/2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2 = x + w/2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2 = y + h/2
        prediction[:, :, :4] = box_corner[:, :, :4]

        # 新建一个output,基于batch_size大小来定None的数量
        output = [None for _ in range(len(prediction))]
        # 基于batch数量遍历(一张一张图来读)
        for i, image_pred in enumerate(prediction):
            # 对种类预测部分沿着最后一个维度取max,并保存数据维度。
            # class_conf  [num_anchors, 1]    类别置信度(类别概率)
            # class_pred  [num_anchors, 1]    类别序号
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # 利用置信度进行第一轮筛选(置信度*分类概率),超过置信度阈值作为mask来筛选符合要求的候选框
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # 根据置信度进行预测结果的筛选
            image_pred = image_pred[conf_mask]  # 所有结果
            class_conf = class_conf[conf_mask]  # 类别置信度
            class_pred = class_pred[conf_mask]  # 类别序号
            # 如果image_pred 0维度的数据大小为0,直接执行continue
            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf(物体置信度), class_conf(类别置信度), class_pred(类别序号)
            # -------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # 获得预测结果中包含的所有种类
            unique_labels = detections[:, -1].cpu().unique()

            # 转化为cuda格式
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            # 遍历每个类别序号
            for c in unique_labels:
                # 获得某一类(同一类别),且通过置信度阈值筛选后全部的预测结果
                detections_class = detections[detections[:, -1] == c]

                # 使用官方自带的非极大抑制会速度更快一些！
                # 筛选出一定区域内,属于同一种类得分最大的框,得到基于分数排序的序号
                keep = nms(
                    detections_class[:, :4],  # box的(x1,y1,x2,y2)
                    detections_class[:, 4] * detections_class[:, 5],  # 拿到物体置信度*分类置信度分数(之后会用来按照分数高低来排序,然后进行筛选,保留分数高的那个)
                    nms_thres  # 非极大值抑制的阈值
                )
                # 经过非极大值抑制之后的框
                max_detections = detections_class[keep]

                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # Add max detections to outputs
                # 将结果加入到output中
                #  如果当前位置没有东西就赋值(类别),如果已经有东西(已经存在类别)则直接拼接concat,相当于一个一个类别拼上去
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            # 循环结束后,如果数据不为空,则进行数据处理
            if output[i] is not None:
                # 转为numpy格式
                output[i] = output[i].cpu().numpy()
                # 格式转换(x1 y1 x2 y2)-->(x y w h)
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
                # y1 x1 y2 x2（还原到原图大小的shape）
        return output

    """
    非极大值抑制(跨类别版本)
    """
    def non_max_suppression_all(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                                nms_thres=0.4):
        """
        :param prediction:预测结果outputs沿着维度1拼接(batch_size, num_anchors, 85),框(x y w h)数据已经归一化
        :param num_classes:类别数量
        :param input_shape:输入网络的shape [640 640]
        :param image_shape:图片的实际高宽 [H W]
        :param letterbox_image:是否通过letter_box的方式来缩放图片到指定尺寸(会有灰色的边),否则就是直接使用resize的方法
        :param conf_thres:置信度阈值
        :param nms_thres:非极大值抑制
        :return: 返回的是单张图片预测出来的框的信息组合成的列表，数据格式为列表[None] List[np[num_anchors, 7][num_anchors, 7]],列表的长度为batch_szie的大小
        [y1 x1 y2 x2 obj_conf(物体置信度)  class_conf(类别置信度)  class_pred(类别序号)]
        """
        # 将预测结果的格式转换成左上角右下角的格式。[x,y,w,h]-->[x1,y1,x2,y2]
        # prediction  [batch_size, num_anchors, 85]
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1 = x - w/2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1 = y - h/2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2 = x + w/2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2 = y + h/2
        prediction[:, :, :4] = box_corner[:, :, :4]  # 把前面的值赋值回去给prediction

        # 新建一个output,基于batch_size大小来定None的数量
        output = [None for _ in range(len(prediction))]
        # 基于batch数量遍历(一张一张图来读)
        for i, image_pred in enumerate(prediction):
            # 对种类预测部分沿着最后一个维度取max,并保存数据维度。
            # class_conf  [num_anchors, 1]    类别置信度(类别概率)
            # class_pred  [num_anchors, 1]    类别序号
            # TODO: 这里可以改成抑制包括不同类别的
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # 利用置信度进行第一轮筛选(置信度*分类概率),超过置信度阈值作为mask来筛选符合要求的候选框
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # 根据置信度进行预测结果的筛选
            image_pred = image_pred[conf_mask]  # 所有结果
            class_conf = class_conf[conf_mask]  # 类别置信度
            class_pred = class_pred[conf_mask]  # 类别序号
            # 如果image_pred 0维度的数据大小为0,直接执行continue
            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf(物体置信度), class_conf(类别置信度), class_pred(类别序号)
            # -------------------------------------------------------------------------#
            # detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            # 在目标跟踪,物体置信度*类别置信度就好
            detections = torch.cat((image_pred[:, :4], image_pred[:, 4:5]*class_conf.float(), class_pred.float()), 1)

            # 获得预测结果中包含的所有种类
            unique_labels = detections[:, -1].cpu().unique()

            # 转化为cuda格式
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            # 不考虑类别
            # 使用官方自带的非极大抑制会速度更快一些！
            # 筛选出一定区域内,属于同一种类得分最大的框,得到基于分数排序的序号
            keep = nms(
                detections[:, :4],  # box的(x1,y1,x2,y2)
                detections[:, 4],  # 拿到物体置信度*分类置信度分数(之后会用来按照分数高低来排序,然后进行筛选,保留分数高的那个)
                nms_thres  # 非极大值抑制的阈值
            )
            # 经过非极大值抑制之后的框
            max_detections = detections[keep]

            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            # 将结果加入到output中
            #  如果当前位置没有东西就赋值(类别),如果已经有东西(已经存在类别)则直接拼接concat,相当于一个一个类别拼上去
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            # 循环结束后,如果数据不为空,则进行数据处理
            if output[i] is not None:
                # 转为numpy格式
                output[i] = output[i].cpu().numpy()
                # 格式转换(x1 y1 x2 y2)-->(x y w h)
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                # y1 x1 y2 x2（还原到原图大小的shape）
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    # ---------------------------------------------------#
    #   将预测值的每个特征层调成真实值
    # ---------------------------------------------------#
    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
        # -----------------------------------------------#
        #   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
        # -----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # -----------------------------------------------#
        #   输入为640x640时 input_shape = [640, 640]  input_height = 20, input_width = 20
        #   640 / 20 = 32
        #   stride_h = stride_w = 32
        # -----------------------------------------------#
        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #   anchor_width, anchor_height / stride_h, stride_w
        # -------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          anchors[anchors_mask[2]]]

        # -----------------------------------------------#
        #   batch_size, 3 * (4 + 1 + num_classes), 20, 20 => 
        #   batch_size, 3, 5 + num_classes, 20, 20  => 
        #   batch_size, 3, 20, 20, 4 + 1 + num_classes
        # -----------------------------------------------#
        prediction = input.view(batch_size, len(anchors_mask[2]),
                                num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # -----------------------------------------------#
        #   先验框的中心位置的调整参数
        # -----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # -----------------------------------------------#
        #   先验框的宽高调整参数
        # -----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # -----------------------------------------------#
        #   获得置信度，是否有物体 0 - 1
        # -----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        # -----------------------------------------------#
        #   种类置信度 0 - 1
        # -----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # ----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角 
        #   batch_size,3,20,20
        #   range(20)
        #   [
        #       [0, 1, 2, 3 ……, 19], 
        #       [0, 1, 2, 3 ……, 19], 
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ] * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        #   
        #   [
        #       [0, 1, 2, 3 ……, 19], 
        #       [0, 1, 2, 3 ……, 19], 
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ].T * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        # ----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)

        # ----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        # ----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # ----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #   x  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_x
        #   y  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_y
        #   w  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_w
        #   h  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_h 
        # ----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5

        box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32
        box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32
        grid_x = grid_x.cpu().numpy() * 32
        grid_y = grid_y.cpu().numpy() * 32
        anchor_w = anchor_w.cpu().numpy() * 32
        anchor_h = anchor_h.cpu().numpy() * 32

        fig = plt.figure()
        ax = fig.add_subplot(121)
        from PIL import Image
        img = Image.open("img/street.jpg").resize([640, 640])
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2

        rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w], anchor_top[0, 0, point_h, point_w]], \
                              anchor_w[0, 0, point_h, point_w], anchor_h[0, 0, point_h, point_w], color="r", fill=False)
        rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w], anchor_top[0, 1, point_h, point_w]], \
                              anchor_w[0, 1, point_h, point_w], anchor_h[0, 1, point_h, point_w], color="r", fill=False)
        rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w], anchor_top[0, 2, point_h, point_w]], \
                              anchor_w[0, 2, point_h, point_w], anchor_h[0, 2, point_h, point_w], color="r", fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
        plt.gca().invert_yaxis()

        pre_left = box_xy[..., 0] - box_wh[..., 0] / 2
        pre_top = box_xy[..., 1] - box_wh[..., 1] / 2

        rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]], \
                              box_wh[0, 0, point_h, point_w, 0], box_wh[0, 0, point_h, point_w, 1], color="r",
                              fill=False)
        rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]], \
                              box_wh[0, 1, point_h, point_w, 0], box_wh[0, 1, point_h, point_w, 1], color="r",
                              fill=False)
        rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]], \
                              box_wh[0, 2, point_h, point_w, 0], box_wh[0, 2, point_h, point_w, 1], color="r",
                              fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #


    feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)
