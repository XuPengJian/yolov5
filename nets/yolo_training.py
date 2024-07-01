import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0):
        super(YOLOLoss, self).__init__()
        # -----------------------------------------------------------#
        #   大目标: 20x20的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   中目标: 40x40的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   小目标: 80x80的特征层对应的anchor是[10,13],[16,30],[33,23]
        # -----------------------------------------------------------#
        self.anchors = anchors  # 获取所有候选框数据,数据格式为[[w1 h1] [w2 h2]...[w9 h9]](默认从小到大排序)
        self.num_classes = num_classes  # 类型数量
        self.bbox_attrs = 5 + num_classes  # bounding box的属性数量 x,y,a,h,置信度confi,classes
        self.input_shape = input_shape  # 输入尺寸 比如[640 640]
        self.anchors_mask = anchors_mask  # 候选框选取依据 [[大] [中] [小]]
        self.label_smoothing = label_smoothing  # 标签平滑,让独热编码部分的值不是完全0or1。一般设置0.01以下。如0.01、0.005。防止过拟合,增加模型的泛化性

        # gt和候选框差距最大的边比值的阈值
        self.threshold = 4

        # loss相关权重系数,比值确定(补偿系数)
        self.balance = [0.4, 1.0, 4]  # 对不同不同特征图给的权重倍数有所不同[依次为大,中,小目标],小目标比较小,所以权重更大
        self.box_ratio = 0.05  # 框损失的权重系数
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)  # 置信度损失的权重系数(基于input_shape大小有关)
        self.cls_ratio = 0.5 * (num_classes / 80)  # 类别损失的权重系数(与类别数量有关)
        self.cuda = cuda  # 使用设备

    # 切分tensor,相当于把tensor的值限制在t_min和t_max之间,小于t_min的就取t_min,大于t_max的就取t_max
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    # BCELoss的定义,包含梯度裁剪的功能,需要置信度来确定这个分类是否符合是个物体的最基本判断
    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)  # 将预测结果限制在epsilon和1-epsilon之间
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    # 计算两个框的iou
    def box_giou(self, b1, b2):
        """
        :param b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh 预测框
        :param b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh 真实框
        :return: giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """

        # 求出预测框左上角右下角
        b1_xy = b1[..., :2]  # 获取预测框中心点xy数值
        b1_wh = b1[..., 2:4]  # 获取预测框wh数值
        b1_wh_half = b1_wh / 2.  # wh的值取一半
        b1_mins = b1_xy - b1_wh_half  # 获得预测框左上角的点的坐标
        b1_maxes = b1_xy + b1_wh_half  # 获预测框得右上角的点的坐标

        # 求出真实框左上角右下角
        b2_xy = b2[..., :2]  # 获取真实框中心点xy数值
        b2_wh = b2[..., 2:4]  # 获取真实框wh数值
        b2_wh_half = b2_wh / 2.  # wh的值取一半
        b2_mins = b2_xy - b2_wh_half  # 获得左上角的点的坐标
        b2_maxes = b2_xy + b2_wh_half  # 获得右上角的点的坐标

        # 求真实框和预测框所有的iou
        intersect_mins = torch.max(b1_mins, b2_mins)  # 左上角的点取最大值
        intersect_maxes = torch.min(b1_maxes, b2_maxes)  # 右下角的点取最小值
        # 得到相交部分的宽高,这里要通过与0比取最大值是为了防止框不存在时,左上角点最大值为一个数,右下角点最小值为0,防止右下-左上得到的时负数
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # 得到相交部分的面积,w*h
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # 计算预测框的面积
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # 计算实际框的面积
        union_area = b1_area + b2_area - intersect_area  # 预测框和实际框并集的面积
        iou = intersect_area / union_area  # 求出iou比值,交集比并集

        # 找到包裹两个框的最小框的左上角和右下角
        enclose_mins = torch.min(b1_mins, b2_mins)  # 左上角的点取最小值
        enclose_maxes = torch.max(b1_maxes, b2_maxes)  # 右下角的点取最大值
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))  # 通过同样方式得到宽高
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]  # 计算得到包裹两个框的最小框的面积
        giou = iou - (enclose_area - union_area) / enclose_area  # 计算giou拿来算loss(直接算iou会有缺点)

        return giou

    # 基于类别数量平滑标签(提高模型的泛化性,使得模型不过度"自信"),类别越多后面补的值越小.相当于把label_smoothing的概率分到了其他类别上
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    # 实例化后默认传入的参数,在计算loss的时候要遍历每个输出头的特征图
    def forward(self, l, input, targets=None, y_true=None):
        # -----------------------------------------------------------------------------#
        # prediction网络输出的值:
        #   l              代表使用的是第几个有效特征层,比如说是三个头的.
        #                  后面将网络输出的三个结果遍历一遍([大0 中1 小2]),l为所在特征层的序号
        #   input的shape为  bs, 3*(5+num_classes), 20, 20
        #                   bs, 3*(5+num_classes), 40, 40
        #                   bs, 3*(5+num_classes), 80, 80
        # gt值:
        #   targets         真实框的标签情况 [batch_size, num_gt, 5]
        #   y_true
        # ------------------------------------------------------------------------------#

        bs = input.size(0)  # batch_size大小,相当于图片数量
        in_h = input.size(2)  # 特征层高h,比如20
        in_w = input.size(3)  # 特征层宽w,比如20
        # -----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   [640, 640] 高的步长为640 / 20 = 32，宽的步长为640 / 20 = 32
        #   如果特征层为20x20的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为40x40的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为80x80的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        # -----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h  # 每一个grid_h对应原来的图片上多少个像素点(或者理解成缩小了多少倍)
        stride_w = self.input_shape[1] / in_w  # 每一个grid_w对应原来的图片上多少个像素点(或者理解成缩小了多少倍)
        # 获得的scaled_anchors大小是相对于特征层的
        # 获得相对于特征层缩放后的框的大小,这里现在有9个框,之后要基于anchor_mask拿出对应特征层的框
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # 将输入到yololoss的input数据格式进行处理,prediction做如下变化
        # input:(bs,3*(5+num_classes),20,20)--view->(bs,3,5+num_classes,20,20) --permute变成-> (bs,3,20,20,5+num_classes)
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()
        # -------------------------------------------------------------------------------#
        #   输入的input一共有三个,经过以上操作改变它们的shape,便于后面取值并拿来计算loss,他们的shape分别是
        #   batch_size, 3, 20, 20, 5 + num_classes
        #   batch_size, 3, 40, 40, 5 + num_classes
        #   batch_size, 3, 80, 80, 5 + num_classes
        # ------------------------------------------------------------------------------#

        # 将网络预测的结果的值都映射到(0,1)之间
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  # x (相对于每个格子左上角来说,中心点x的位置(0-1)之间),之后会映射到(-0.5,1.5),相当于每个格子的影响范围offset0.5
        y = torch.sigmoid(prediction[..., 1])  # y (相对于每个格子左上角来说,中心点y的位置(0-1)之间),之后会映射到(-0.5,1.5),相当于每个格子的影响范围offset0.5
        # 先验框的宽高调整参数
        w = torch.sigmoid(prediction[..., 2])  # w (相对于先验框来说,宽的倍数(0-1)之间),之后会映射到0~4倍
        h = torch.sigmoid(prediction[..., 3])  # h (相对于先验框来说,宽的倍数(0-1)之间),之后会映射到0~4倍
        # 获得预测出的置信度, 是否有物体
        conf = torch.sigmoid(prediction[..., 4])  # conf,映射到(0-1)之间
        # 种类置信度, 预测出的类别概率
        pred_cls = torch.sigmoid(prediction[..., 5:])  # cls
        # -----------------------------------------------#
        # self.get_target已经合并到dataloader中(用于得到y_true)
        # y_true的内容为[[3(anchor_num),20*20(特征层大小),5+cls],[3(anchor_num),40*40(特征层大小),5+cls],[3(anchor_num),60*60(特征层大小),5+cls]]
        # 原因是在这里执行过慢，会大大延长训练时间
        # -----------------------------------------------#
        # y_true, noobj_mask = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        # 基于特征层格子数量,候选框大小.获取网络输出结果映射到框上的数据
        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)
        # target的格式为(bs,box_num(gt框的数量),5(x,y,w,h,cls)),并且是归一化x,y,w,h到0-1之间.外面的总数是代表每张图框的数量

        if self.cuda:
            y_true = y_true.type_as(x)  # 将y_true的数据类型与x设置为一致.毕竟基本上都是0,1,以免是int类型

        # 初始化loss值,准备开始计算loss
        loss = 0.
        loss_loc = 0.
        loss_cls = 0.
        # 储存各部分的loss [box, cls, conf]
        losses = torch.zeros(3, requires_grad=False)
        # 求gt中所有存在置信度的数量
        n = torch.sum(y_true[..., 4] == 1)
        # 如果存在物体(gt但凡有一个格子拥有置信度,也就是gt有框的前提下)
        if n != 0:
            # 计算预测结果和真实结果的giou,计算对应有真实框的先验框的giou损失
            # loss_cls计算对应有真实框的先验框的分类损失,对box算iou损失
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            # 计算框损失1-giou,并且要求置信度满足conf==1的情况的值参与计算(相当于L1loss)
            loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])

            # 计算分类损失,并要求置信度满足conf==1,(BCEloss二分类损失,使得可以完成多分类任务)
            # 这里对target做了标签平滑,防止过拟合
            loss_cls = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1],
                                               self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1],
                                                                  self.label_smoothing, self.num_classes)))
            # 乘上补偿系数
            loss_loc *= self.box_ratio
            loss_cls *= self.cls_ratio

            # 合并计算损失
            loss += loss_loc  # 位置损失
            loss += loss_cls  # 类别损失

            # 计算置信度的loss
            # 也就意味着先验框对应的预测框预测地更准确
            # 它才是用来预测这个物体的。最小值做限制在0
            tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        # 如果gt完全不存在物体,直接gt置信度全赋予0(建个全0列表)
        else:
            tobj = torch.zeros_like(y_true[..., 4])

        # 置信度损失(BCELoss计算)
        loss_conf = torch.mean(self.BCELoss(conf, tobj))

        # 最终计算地loss再加上这一项置信度误差,带上相应的系数(根据不同特征层也赋予不同的系数)
        loss_conf = loss_conf * self.balance[l] * self.obj_ratio

        loss += loss_conf  # 置信度损失

        # 结果存放到losses中
        losses[0] = (loss_loc)  # box_loss
        losses[1] = (loss_cls)  # cls_loss
        losses[2] = (loss_conf)  # conf_loss
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss, losses

    # 基于特征层格子数量,候选框大小.获取网络输出结果映射到框上的数据
    def get_pred_boxes(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w):
        # 计算一共有多少张图片,其实相当于一个batch的大小,batch_size
        bs = len(targets)

        # 生成网格，先验框中心，即网格左上角
        # 这里的.view(x.shape)是为了让其和网络的输出处理后的格式一致,都为(batch_size, 3, 20, 20)  最后一个维度后面会进行unsqueeze再concat
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)  # 网格的x坐标左上角点水平方向(0-19)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)  # 网格的y坐标左上角点水平方向(0-19)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]  # 拿出所在特征图对应的三个先验框
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)  # 获得缩放后先验框的w数据
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)  # 获得缩放后先验框的h数据
        # 将先验框的数据拷贝到每个batch个数和网格之中,让其和网络的输出处理后的格式一致,都为(batch_size, 3, 20, 20)
        # [[w1 w2 w3]](简写成[[...]])-第一次repeat(bs维度拷贝)->[[...][...]]-第二次repeat(grid_size维度拷贝)->[[... ...][... ...]]
        # 然后再reshape成(batch_size, 3(在这里是三个不同尺寸的候选框数据,刚好对应w1,w2,w3), 20, 20)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与宽高(映射到网络中能对应上的值)
        # x: 0~1,   x * 2. : 0~2,    x * 2. - 0.5: -0.5~1.5 ,因为考虑了匹配相邻的点(三个),所以值域范围变成了-0.5~1.5(左右各多了0.5范围)
        # w: 0~1,   (w * 2): 0~2,    (w * 2) ** 2: 0~4  ,与之前设置的threshold=4相关,把结果倍数限制在(0-4)之间
        # 统一在最后面增加一个维度,用于拼接
        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)  # 这里的grid_x是网格的x坐标序号
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)  # 这里的grid_y是网格的y坐标序号
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)  # anchor_w是候选框缩放到此网格中的框w大小
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)  # anchor_h是候选框缩放到此网格中的框h大小
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)  # 将结果拼接起来
        return pred_boxes


# 判断模型是否使用多GPU平行分布式训练策略,用于ema判断训练方式
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


# 如果是平行策略则直接返回model.module, 如果是不是分布式训练直接返回model
def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


# 拷贝参数,可以通过include和exclude包含或排除一些值
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        # 如果设置了include参数且k不在include里面,或者,k是私有属性,或者,k在exclude内,继续选下个k值进行循环
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        # 否则就设置新的参数
        else:
            setattr(a, k, v)  # object,name,vale


# ema是用来在test或val时才使用，用来更新参数
class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        # 衰减指数平滑,用于帮助早期的epoch找到更新方向
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    # 更新ema模型参数
    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    # 更新属性方法(通过拷贝属性的方式)
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


# 权重随机初始化(在没有使用预训练模型的时候使用)
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# scheduler学习率下降的公式(实例化scheduler学习率调度器,学习率变化策略)
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
                                              ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


# 设置优化器学习率调整策略
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
