import datetime
import random
import math
import os
import glob
import sys

import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import download_weights, get_anchors, get_classes, show_config, draw_boxes, get_lr, load_pretrained_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''

'''
训练超参设置
'''


def parse_args():
    parser = argparse.ArgumentParser()
    # 有关数据集的读取
    parser.add_argument('--classes_path', type=str, default='model_data/cars_classes.txt',
                        help='存放类别的txt文件路径')
    parser.add_argument('--anchors_path', type=str, default='model_data/yolo_anchors.txt',
                        help='存放先验框信息的txt文件路径')
    parser.add_argument('--train_annotation_path', type=str, default='preprocess/cars2_train.txt',
                        help='训练集的txt文件路径')
    parser.add_argument('--val_annotation_path', type=str, default='preprocess/cars2_val.txt',
                        help='验证集的txt文件路径')

    # 模型训练参数
    parser.add_argument('--input_shape', type=list, default=[864, 1536],  # [1152, 2048]
                        help='网络的输入分辨率大小[h, w],一定要为32的倍数')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='bs')
    parser.add_argument('--drop_last', type=bool, default=True,
                        help='是否丢弃最后不满足batch_size大小的部分')

    # 预训练模型配置
    parser.add_argument('--phi', type=str, default='s', help='预训练模型使用的版本')
    parser.add_argument('--pretrained', type=str,
                        default=r'E:\gitlab\cars_detection\yolov5\train_result\best_weights_small2.pth',
                        # 'model_data/yolov5_s_v6.1.pth'
                        help='预训练权重路径')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的文件夹路径，里面应该包含model.pt和optimizer.pt')

    parser.add_argument('--amp', type=bool, default=True,
                        help='是否使用混合精度训练')
    # parser.add_argument('--warmup_epoch', type=int, default=5,
    #                     help='在训练开始时预热几个epoch，-1就是不warmup')

    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--save_period', type=int, default=1,
                        help='每多少个epoch保存一次训练结果')
    parser.add_argument('--early_stop_epoch', type=int, default=50,
                        help='早停策略，当连续这么多epoch loss不下降的时候，训练停止')
    parser.add_argument('--seed', type=int, default=20230209,
                        help='训练时的随机种子，如果设为-1则随机')

    # 学习率相关
    parser.add_argument('--init_lr', type=float, default=1e-2,
                        help='模型初始化学习率')
    parser.add_argument('--min_lr', type=float, default=1e-2 * 0.01,
                        help='模型最小学习率')
    parser.add_argument('--optimizer_type', type=str, default='sgd',
                        help='当使用adam优化器时建议设置  Init_lr=1e-3，当使用SGD优化器时建议设置Init_lr=1e-2')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='优化器内部使用到的momentum参数')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权值衰减，可防止过拟合，adam会导致weight_decay错误，使用adam时建议设置为0,sgd的default=5e-4')
    parser.add_argument('--lr_decay_type', type=str, default='cos',
                        help='使用到的学习率下降方式，可选的有step、cos')

    # 有关数据增强
    parser.add_argument('--mosaic', type=bool, default=True,
                        help='使用mosaic数据增强')
    parser.add_argument('--mosaic_prob', type=float, default=0.5,
                        help='mosaic数据增强的概率，默认每个step50%的概率')
    parser.add_argument('--mixup', type=bool, default=True, help='使用mixup数据增强')
    parser.add_argument('--mixup_prob', type=float, default=0.5,
                        help='mixup数据增强的概率，默认每个step50%的概率')
    parser.add_argument('--special_aug_ratio', type=float, default=0.7,
                        help='默认前70%个epoch使用Mosaic数据增强，后面30%则不使用')
    parser.add_argument('--label_smoothing', type=float, default=0.005,
                        help='标签平滑。一般0.01以下。如0.01、0.005。这是一种正则化策略')
    parser.add_argument('--letterbox', type=bool, default=False,
                        help='resize图片时是否使用letterbox')

    # 验证集相关
    parser.add_argument('--eval_flag', type=bool, default=True,
                        help='是否在训练时进行验证（验证集），安装pycocotools库后，评估体验更佳')
    parser.add_argument('--eval_period', type=int, default=1,
                        help='多少个eopch评估一次mAP,并且采样一下validation的结果,打印图片')  # sample_interval
    parser.add_argument('--num_workers', type=int, default=0,
                        help='多线程读取数据，加快读取速度，但是会更占用内存')

    return parser.parse_args()


'''
训练主函数
'''


def main(args):
    Cuda = True  # 是否使用cuda进行训练

    # 设置随机种子
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # 获取当前时间，创建本次训练的目录
    train_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_start_time = train_start_time.replace(' ', '_').replace('-', '_').replace(':', '_')
    save_folder = os.path.join('train_result', train_start_time)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # 创建log保存文件夹
    log_folder = os.path.join(save_folder, 'Log')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # 创建sample的图片保存路径
    image_sample_folder = os.path.join(log_folder, 'sample_images')
    if not os.path.exists(image_sample_folder):
        os.makedirs(image_sample_folder)
    # 创建权重保存文件夹
    model_save_folder = os.path.join(save_folder, 'Model')
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    # 设置训练的device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO:这个东西加上有什么用
    if str(device) == 'cuda':
        # cuDNN是英伟达专门为深度神经网络所开发出来的GPU加速库，针对卷积、池化等等常见操作做了非常多的底层优化，比一般的GPU程序要快很多。
        # 在使用cuDNN的时候，默认为False。设置为True将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
        # 进而实现网络的加速。适用场景是网络结构固定，网络输入形状不变（即一般情况下都适用）。反之，如果卷积层的设置一直变化，
        # 将会导致程序不停地做优化，反而会耗费更多的时间。
        torch.backends.cudnn.benchmark = True
        # 如果要复现结果的话，需要将下面的flag设置为True，这会保证每次返回的卷积算法是确定的，也就是默认算法。
        # torch.backends.cudnn.deterministic = True

    # 获取classes, anchor(候选框)
    class_names, num_classes = get_classes(args.classes_path)  # 获得类别名称(按顺序的列表)和类别数量
    anchors, num_anchors = get_anchors(args.anchors_path)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # txt是从小到大排序，这里的mask是从大目标到小目标从小到大的排序

    # TODO:这里先做检查然后再考虑是否下载
    # 下载预训练模型 # cspdarknet_s_v6.1_backbone.pth
    # if args.pretrained:
    #     # TODO:这个phi可不可以读文件名直接得到?
    #     download_weights(args.phi)

    # 实例化yolo模型, 这里的pretrained设置为False其实是官方的backbone预训练权重的下载,一般不使用
    model = YoloBody(anchors_mask, num_classes, args.phi, pretrained=False)

    # 获得损失函数
    yolo_loss = YOLOLoss(anchors, num_classes, args.input_shape, Cuda, anchors_mask, args.label_smoothing)

    # 记录loss
    # time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(log_folder, "loss")  # 损失保存文件夹路径
    loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)

    # 使用混合精度训练
    if args.amp:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    # 模型训练模式
    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 权值平滑(# ema是用来在test或val时才使用，用来更新参数)
    ema = ModelEMA(model_train)

    # 读取数据集对应的txt
    with open(args.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 打印数据集数量
    print('Number of train images: {};    Number of val images: {}'.format(num_train, num_val))

    # 展示所有数据信息
    # TODO:这里需要大改?
    show_config(
        classes_path=args.classes_path, anchors_path=args.anchors_path, anchors_mask=anchors_mask,
        model_path=args.pretrained,
        input_shape=args.input_shape, Init_Epoch=0, Freeze_Epoch=0, UnFreeze_Epoch=args.epochs,
        Freeze_batch_size=0, Unfreeze_batch_size=args.batch_size, Freeze_Train=False, \
        Init_lr=args.init_lr, Min_lr=args.min_lr, optimizer_type=args.optimizer_type, momentum=args.momentum,
        lr_decay_type=args.lr_decay_type, save_period=args.save_period, save_dir='train_result',
        num_workers=args.num_workers,
        num_train=num_train, num_val=num_val
    )

    # 计算每个epoch里step的长度,仅用来判断数据集的长度,是否读取到数据集(后面还会根据drop_last做更精确的计算)
    epoch_step = num_train // args.batch_size if args.drop_last else math.ceil(num_train / args.batch_size)
    epoch_step_val = num_val // args.batch_size if args.drop_last else math.ceil(num_val / args.batch_size)
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # 一些batch_size推荐值
    wanted_step = 5e4 if args.optimizer_type == "sgd" else 1.5e4
    total_step = num_train // args.batch_size * args.epochs if args.drop_last else math.ceil(
        num_train / args.batch_size) * args.epochs
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // args.batch_size) + 1
        print(
            f"\n\033[1;33;44m[Warning] 使用{args.optimizer_type}优化器时，建议将训练总步长设置到{wanted_step}以上。\033[0m")
        print(
            f"\033[1;33;44m[Warning] 本次运行的总训练数据量为{num_train}，batch_size为{args.batch_size}，共训练{args.epochs}个Epoch，计算出总训练步长为{total_step}。\033[0m")
        print(
            f"\033[1;33;44m[Warning] 由于总训练步长为{total_step}，小于建议总步长{wanted_step}，建议设置总世代为{wanted_epoch}。\033[0m")

    # 判断当前batch_size,自适应调整学习率
    nbs = 64
    lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(args.batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(args.batch_size / nbs * args.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 根据optimizer_type选择优化器
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():  # 返回包含所有子模块（直接、间接）的迭代器，同时产生模块的名称以及模块本身
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    # 建立一个optimizer优化器的字典,通过optimizer_type选择对应的优化器
    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(args.momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=args.momentum, nesterov=True)
    }[args.optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": args.weight_decay})  # weight 需要使用权重衰减(adam一般设置为0)
    optimizer.add_param_group({"params": pg2})  # bias 不需要使用权重衰减

    # 使用权值平滑
    # if ema:
    #     ema.updates = epoch_step * Init_Epoch

    # 构建dataset(单张图片的处理策略)
    train_dataset = YoloDataset(train_lines, args.input_shape, num_classes, anchors, anchors_mask,
                                epoch_length=args.epochs, mosaic=args.mosaic, mixup=args.mixup,
                                mosaic_prob=args.mosaic_prob, mixup_prob=args.mixup_prob, train=True,
                                special_aug_ratio=args.special_aug_ratio)
    val_dataset = YoloDataset(val_lines, args.input_shape, num_classes, anchors, anchors_mask, epoch_length=args.epochs,
                              mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

    # 构建dataloader(batch的划分与处理策略)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=args.drop_last, collate_fn=yolo_dataset_collate, sampler=None)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=args.drop_last, collate_fn=yolo_dataset_collate, sampler=None)

    # 记录eval的map曲线
    eval_callback = EvalCallback(model, args.input_shape, anchors, anchors_mask, class_names, num_classes,
                                 val_lines, log_folder, Cuda, eval_flag=args.eval_flag, period=args.eval_period,
                                 letterbox_image=args.letterbox)

    # 加载resume模型
    if args.resume != '':
        model_path = glob.glob(os.path.join(args.resume, 'last_weights.pth'))[0]
        opt_path = glob.glob(os.path.join(args.resume, 'last_optimizer.pth'))[0]
        load_pretrained_model(model, model_path)  # 加载预训练模型
        optimizer.load_state_dict(torch.load(opt_path))  # 加载预训练学习率
        print('\nResume successfully!')

    # 加载pretrained预训练模型的权重,根据预训练权重的Key和模型的Key进行加载
    elif args.pretrained != '':
        # 这里是之前的写法:
        # model_dict = model.state_dict()
        # pretrained_dict = torch.load(args.pretrained, map_location='cpu')
        # load_key, no_load_key, temp_dict = [], [], {}
        # for k, v in pretrained_dict.items():
        #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        #         temp_dict[k] = v
        #         load_key.append(k)
        #     else:
        #         no_load_key.append(k)
        # model_dict.update(temp_dict)
        # model.load_state_dict(model_dict)
        # 显示没有匹配上的key
        # print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        # print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

        load_pretrained_model(model, args.pretrained)
        print(f'成功加载{args.pretrained}预训练模型!')

    # 如果没使用预训练模型
    else:
        weights_init(model)  # 使用权重初始化方法

    # 获取当前显卡的显存
    mem_total = f'{0:.3g}G'
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device)
        mem_total = '{:.3g}G'.format(p.total_memory / 1E9)

    """
    开始训练
    """
    # 早停策略初始化参数
    early_stop_now = 0  # 当前epoch
    early_stop_epoch = args.early_stop_epoch  # loss不再下降的epoch数

    # 开始每个epoch进行循环
    for epoch in range(args.epochs):
        # 判断当前的batch_size, 自适应调整学习率
        nbs = 64  # nominal batch size  名义批次(固定为64的batch_size更新一次梯度,可以看成梯度累加)
        # 限制最小的learning_rate不要变得过于小,最大的learning_rate不要变得过于大
        lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
        # 每一份batch如果不满足64的大小,基于一定比例进行梯度累加(这里是限制了在每64的倍数的batch_size里,learning_rate的均等分)
        Init_lr_fit = min(max(args.batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(args.batch_size / nbs * args.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # 实例化scheduler学习率下降的公式(实例化scheduler学习率调度器,学习率变化策略)
        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)

        # 让 backward 可以追踪这个参数并且计算它的梯度
        # 当你在使用 Pytorch 的 nn.Module 建立网络时，其内部的参数都自动的设置为了 requires_grad=True，故可以直接取梯度
        for param in model.backbone.parameters():
            param.requires_grad = True

        # 这里根据是否drop_last来决定实际epoch_step的数量
        epoch_step = num_train // args.batch_size if args.drop_last else math.ceil(num_train / args.batch_size)
        epoch_step_val = num_val // args.batch_size if args.drop_last else math.ceil(num_val / args.batch_size)

        # 获取ema更新的step总数
        if ema:
            ema.updates = epoch_step * epoch

        # 在dataset中更新当前epoch的属性,主要用于判断是否还使用mosaic和mixup数据增强
        train_dataloader.dataset.epoch_now = epoch
        val_dataloader.dataset.epoch_now = epoch

        # 设置优化器学习率调整策略
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # 初始化损失函数(对于整个epoch的,用于累加)
        loss = 0
        avg_loss = 0  # 用与储存当前step之前平均的loss
        avg_losses = torch.zeros(3, requires_grad=False)
        losses = torch.zeros(3, requires_grad=False)
        val_loss = 0
        val_avg_loss = 0  # 用与储存当前step之前平均的loss
        val_avg_losses = torch.zeros(3, requires_grad=False)
        val_losses = torch.zeros(3, requires_grad=False)

        """
        训练阶段
        """

        print('\nStart Train...')
        # 训练过程可视化成进度条
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3)
        model_train.train()

        # 训练阶段:打印具体表头信息
        print('{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}'.format('Epoch', 'gpu_mem', 'loss_overall',
                                                                              'box_loss', 'cls_loss', 'conf_loss',
                                                                              'lr', 'bs', 'img_size'))
        n = 0  # 计算batch实际计算个数(用于计算平均loss,排除掉NAN的情况,极端图)
        for iteration, batch in enumerate(train_dataloader):
            n += 1
            # iteration其实就是一个epoch中batch的数量（也就是梯度更新的次数）
            if iteration >= epoch_step:
                break
            # 读取图片和标签数据,并放在显存中
            images, targets, y_trues = batch[0], batch[1], batch[2]
            # # TODO:尝试打印图片看一下
            # if iteration % 200 == 0:
            #     sample_num += 1
            #     img = torch.squeeze(batch[0][0], dim=0).permute(1, 2, 0).cpu().numpy()
            #     img = (img[:, :, ::-1] * 255).astype(np.uint8)
            #     cv2.imwrite(f'{save_folder}/epoch_{epoch + 1}_{sample_num}.png', img)

            with torch.no_grad():
                images = images.to(device)
                targets = [target.to(device) for target in targets]
                y_trues = [y_true.to(device) for y_true in y_trues]

            optimizer.zero_grad()  # 梯度清零

            if not args.amp:
                outputs = model_train(images)  # 前向传播
                loss_value_all = 0
                losses_value_all = torch.zeros(3, requires_grad=False)
                # 计算损失
                for l in range(len(outputs)):
                    loss_item, losses_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                    loss_value_all += loss_item
                    losses_value_all += losses_item
                loss_value_all.backward()  # 反向传播
                optimizer.step()  # 梯度更新

            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model_train(images)  # 前向传播
                    loss_value_all = 0
                    losses_value_all = torch.zeros(3, requires_grad=False)
                    # 计算损失
                    for l in range(len(outputs)):
                        loss_item, losses_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                        loss_value_all += loss_item
                        losses_value_all += losses_item
                    # 如果loss出现NAN数据则直接跳过当前的batch
                    if math.isnan(loss_value_all):
                        n -= 1
                        print(f"存在loss为NAN值,已跳过当前epoch第{iteration+1}个batch")
                        continue

                scaler.scale(loss_value_all).backward()  # 反向传播
                scaler.step(optimizer)
                scaler.update()

            # 通过指数滑动平均更新参数
            if ema:
                ema.update(model_train)

            # 求一个epoch里loss求和(用于显示)
            loss += loss_value_all.item()  # 整个epoch的loss累加,之后要除以(interation+1),因为要考虑NAN,所以用n来计算batch计算数量
            losses += losses_value_all

            # 求一个epoch里已经训练的step的平均(用于显示)
            avg_loss = loss / n
            avg_losses = losses / n
            box_loss = avg_losses[0].item()
            cls_loss = avg_losses[1].item()
            conf_loss = avg_losses[2].item()

            # 进度条后缀,训练过程可视化每个step平均的loss和lr的变化
            # pbar.set_postfix(**{'loss': loss, 'lr': get_lr(optimizer)})

            # 反馈当前进程中Torch.Tensor所占用的GPU显存
            mem = f'{torch.cuda.max_memory_allocated(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            current_lr = optimizer.param_groups[0]['lr']
            img_size = '[' + str(images.shape[-2:][0]) + ' ' + str(images.shape[-2:][1]) + ']'

            pbar.set_description('{:^15}{:^15}{:^15.6f}{:^15.6f}{:^15.6f}{:^15.6f}{:^15.6f}{:^15}{:^15}'.format(
                f'{epoch + 1}/{args.epochs}', f'{mem}/{mem_total}', avg_loss, box_loss, cls_loss, conf_loss, current_lr,
                len(targets), img_size))

            pbar.update(1)

        pbar.close()
        print('Finish Train\n')

        """
        验证阶段
        """
        print('Start Validation...')
        # 验证阶段:打印具体表头信息
        print('{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}'.format('Epoch', 'gpu_mem', 'loss_overall',
                                                                              'box_loss', 'cls_loss', 'conf_loss',
                                                                              'lr', 'bs', 'img_size'))

        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3)

        if ema:
            model_train_eval = ema.ema
        else:
            model_train_eval = model_train.eval()

        for iteration, batch in enumerate(val_dataloader):
            if iteration >= epoch_step_val:
                break
            images, targets, y_trues = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images = images.to(device)
                targets = [target.to(device) for target in targets]
                y_trues = [y_true.to(device) for y_true in y_trues]

                optimizer.zero_grad()  # 梯度清零
                outputs = model_train_eval(images)  # 前向传播
                loss_value_all = 0
                losses_value_all = torch.zeros(3, requires_grad=False)
                # 计算损失
                for l in range(len(outputs)):
                    loss_item, losses_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                    loss_value_all += loss_item
                    losses_value_all += losses_item
                loss_value_all = loss_value_all
                losses_value_all = losses_value_all

            # epoch损失求和(用于显示)
            val_loss += loss_value_all.item()
            val_losses += losses_value_all

            # epoch损失求平均(用于显示)
            val_avg_loss = val_loss / (iteration + 1)
            val_avg_losses = val_losses / (iteration + 1)
            val_box_loss = val_avg_losses[0].item()
            val_cls_loss = val_avg_losses[1].item()
            val_conf_loss = val_avg_losses[2].item()

            # 进度条后缀
            # pbar.set_postfix(**{'val_loss': val_loss})

            # 反馈当前进程中Torch.Tensor所占用的GPU显存
            mem = f'{torch.cuda.max_memory_allocated(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            current_lr = optimizer.param_groups[0]['lr']
            img_size = '[' + str(images.shape[-2:][0]) + ' ' + str(images.shape[-2:][1]) + ']'

            pbar.set_description('{:^15}{:^15}{:^15.6f}{:^15.6f}{:^15.6f}{:^15.6f}{:^15.6f}{:^15}{:^15}'.format(
                f'{epoch + 1}/{args.epochs}', f'{mem}/{mem_total}', val_avg_loss, val_box_loss, val_cls_loss,
                val_conf_loss,
                current_lr,
                len(targets), img_size))

            pbar.update(1)

        pbar.close()
        print('Finish Validation\n')

        # 绘制loss下降的图像
        loss_history.append_loss(epoch + 1, avg_loss, val_avg_loss, avg_losses, val_avg_losses)
        # map计算(利用可pycocotools)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(args.epochs))
        print('Total Loss: %.6f || Val Loss: %.6f ' % (avg_loss, val_avg_loss))

        # 每几个epoch采样一下train的结果,打印图片
        if (epoch + 1) % args.eval_period == 0:
            # 采样训练结果为图片
            image_sample_folder_epoch = os.path.join(image_sample_folder, '{}'.format(epoch + 1))
            # 创建给当前epoch存放图片的文件夹
            if not os.path.exists(image_sample_folder_epoch):
                os.makedirs(image_sample_folder_epoch)
            # txt的名称其实映射的就是图片的名称
            det_txts = glob.glob(os.path.join(log_folder, 'map_out_path', 'detection-results', '*.txt'))
            # 从检测结果中随机抽取10张图
            show_list = np.random.choice(det_txts, 10, replace=False)
            # 遍历抽取出来的10张图片
            for txt in show_list:
                # 获得对应的gt信息
                gt_txt = txt.replace('detection-results', 'ground-truth')
                # 打开文件逐行读取信息
                with open(txt, 'r') as f:  # dr
                    boxes_det = f.readlines()
                with open(gt_txt, 'r') as f:  # gt
                    boxes_gt = f.readlines()
                file_name = os.path.basename(txt).split('.')[0]  # 提取文件名(除去后缀部分)
                for line in val_lines:  # 遍历验证集图片的名字(不包括后缀部分)
                    if os.path.basename(line.split()[0]).split('.')[0] == file_name:  # 如果file_name对应上了
                        image_path = os.path.abspath(line.split()[0])  # 获取图片的相对路径
                        image = cv2.imread(image_path)  # cv2打开图片
                        draw_boxes(file_name, image, boxes_det, boxes_gt, image_sample_folder_epoch, class_names,
                                   num_classes)

        # 保存模型权重
        if ema:
            save_state_dict = ema.ema.state_dict()
            save_opt_dict = optimizer.state_dict()
        else:
            save_state_dict = model.state_dict()
            save_opt_dict = optimizer.state_dict()

        if (epoch + 1) % args.save_period == 0 or epoch + 1 == args.epochs:
            torch.save(save_state_dict, os.path.join(model_save_folder, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))  # 保存权重文件
            torch.save(save_opt_dict, os.path.join(model_save_folder, "ep%03d-loss%.3f-val_loss%.3f_opt.pth" % (
                epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))  # 保存学习率文件

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(model_save_folder, "best_weights.pth"))  # 保存权重文件
            torch.save(save_opt_dict, os.path.join(model_save_folder, "best_optimizer.pth"))  # 保存学习率文件

        torch.save(save_state_dict, os.path.join(model_save_folder, "last_weights.pth"))  # 保存权重文件
        torch.save(save_opt_dict, os.path.join(model_save_folder, "last_optimizer.pth"))  # 保存学习率文件

        # 关闭文件写入(这里针对的是SummaryWriter,用于tensorboard的可视化)
        loss_history.writer.close()

        # 检测是否早停
        if val_loss / epoch_step_val <= min(loss_history.val_loss):
            early_stop_now = 0
        else:
            early_stop_now = early_stop_now + 1
            print('\nEarly Stop Epoch [{}<{}]'.format(early_stop_now, early_stop_epoch))
        if early_stop_now >= early_stop_epoch:
            print('\n网络验证部分损失已连续{}个epoch没有下降，触发早停机制，训练结束！'.format(early_stop_epoch))
            sys.exit()

        # 分割线打印
        print('----------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    main(args)
