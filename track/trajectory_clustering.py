import colorsys
from copy import deepcopy

import numpy as np
import math
from math import pi
import cv2
from PIL import Image
import imageio.v2 as iio

import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
from track.QuickBundleClustering import cal_trajectory_length, resample_points, normalize, extend_lines
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import MidpointFeature, VectorOfEndpointsFeature, CenterOfMassFeature
from dipy.segment.metric import EuclideanMetric, CosineMetric
from scipy import stats

"""
轨迹归并算法,得到归并后的结果
"""


def cluster_tracks(txt_path, h, w, threshold=0.125, min_cars=5):
    """
    :param txt_path: 输出output_txt
    :param h: 原始图像的高度
    :param w: 原始图像的宽度
    :return:
    track_dic: 轨迹分类结果的字典形式
    track_cls_dic: 每个轨迹每个车类别对应的数量
    track_representation: 轨迹代表线 list
    track_count_percentage: 每个轨迹数量所占百分比 list

    超参:
    num_points: 点重采样的采样点数量resample_points
    threshold: QuickBundles的阈值范围
    min_points_count: 最少检测点的数量,作为过滤依据
    min_length = 最短检测线长度过滤,作为过滤依据(归一化结果)
    """
    # 超参部分
    min_points_count = 60  # 最少检测点的数量,作为过滤依据
    min_length = 0.3  # 最短检测线长度过滤,作为过滤依据(归一化结果),尽可能要完整长度的

    # 帧, id, x1, y1, x2, y2, conf, cls
    # 3,4,2052,996,2086,1061,0.881012,1
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    trajectory = {}  # 用来记录轨迹的点坐标(组合成线坐标)
    cls_trajectory = {}  # 用来记录不同类别在不同轨迹的数量
    points = []
    w_size_list = []  # 用于记录w的尺寸(归一化结果)
    h_size_list = []  # 用于记录h的尺寸(归一化结果)

    # 遍历txt每一行数据
    for line in lines:
        info_list = line.replace('\n', '').split(',')[0:8]
        frame, id, x1, y1, x2, y2, conf, cls = info_list
        cls = int(cls)
        # 这里面的坐标长度都是经过归一化处理过之后的
        x_center = (float(x1) + float(x2)) / (2 * w)
        y_center = (float(y1) + float(y2)) / (2 * h)
        anchor_w = (float(x2) - float(x1)) / w
        anchor_h = (float(y2) - float(y1)) / h
        if x_center > 1 or x_center < 0 or y_center > 1 or y_center < 0:
            continue
        points.append([x_center, y_center])
        w_size_list.append(anchor_w)
        h_size_list.append(anchor_h)
        # 将同一id的车的中心点xy，写入到轨迹字典中
        if id in trajectory:
            # 将同一id的车的中心点xy
            trajectory[id].append([x_center, y_center])
            # 记录该id对应的类别(这里其实会重复append，所以这里不需要append了，直接在else赋值一个就行)
            # cls_trajectory[id].append(cls)
        else:
            trajectory[id] = [[x_center, y_center]]
            cls_trajectory[id] = cls

    # 获取处于中位数的anchor尺寸,然后取一个最小的
    # w_size_list.sort()
    # h_size_list.sort()
    # middle_w_size = w_size_list[len(w_size_list) // 2]
    # middle_h_size = h_size_list[len(h_size_list) // 2]
    # car_w = min(middle_w_size, middle_h_size)
    # car_h = max(middle_w_size, middle_h_size)
    # # print('w  :', car_w)
    # # print('h  :', car_h)
    # # print('w/h:', car_w/car_h)
    # threshold = car_h * 4.0  # 框的尺寸与threshold的关系,超参
    # threshold = 0.125
    # print(len(trajectory))

    # ------------先筛选点太少和长度太短的轨迹(前期筛选,于执行算法前)------------
    del_keys = []
    # 遍历每个id的轨迹
    for key in trajectory:
        # 计算轨迹长度
        max_length = cal_trajectory_length(trajectory[key])
        # 点数小于阈值或者长度小于阈值的轨迹进行删除
        if len(trajectory[key]) < min_points_count or max_length < min_length:
            # 记录需要删除id的轨迹
            del_keys.append(key)

    # 根据key删除不要的轨迹
    for key in del_keys:
        del trajectory[key]
        del cls_trajectory[key]

    # 因为字典是无序的(其实是按照车辆id来排)，不方便后续索引，所以转成list
    trajectory_list = []  # 轨迹信息
    cls_trajectory_list = []  # 类别信息
    id_trajectory_list = []  # 车辆id信息（保留下来的id，后面也要跟着聚类之后的分类来走）

    # 曲线重采样，一种非线性的重采样，并让点变得均分(第一次)
    for key in trajectory:
        # 在此处resample_points,将点数据转化为numpy格式了
        points_new = resample_points(trajectory[key], num_points=32)
        # ---前期线段延长(基于首尾向量做调整,调整成新的轨迹)---
        points_new = extend_lines(points_new)
        # 得到新的点,放入到适当位置
        trajectory[key] = points_new

    # 再次曲线重采样，因为做了延长操作，所以点又变的不均分了,所以希望让点变得均分(第二次)
    for key in trajectory:
        # 在此处resample_points,将点数据转化为numpy格式了
        points_new = resample_points(trajectory[key], num_points=24)
        # 得到新的点,放入到轨迹字典中
        trajectory[key] = points_new
        # 得到新的点,放入到轨迹列表中
        trajectory_list.append(points_new)
        cls_trajectory_list.append(cls_trajectory[key])
        id_trajectory_list.append(key)

    # trajs = []
    # # 将数据改为三维的，以符合QuickBundle算法需求
    # for key in trajectory:
    #     trajectory_3d = []
    #     for point in trajectory[key]:
    #         point_3d = np.append(point, 0.0)
    #         trajectory_3d.append(point_3d)
    #     trajectory_3d = np.array(trajectory_3d, dtype=np.float32)
    #     trajs.append(np.array(trajectory_3d))

    # -----------------基于位置信息(中点)聚类---------------------
    # 执行QuickBundles算法
    # 选取合适聚类的方法
    # 使用中点特征(位置信息)作为聚类手段
    feature = MidpointFeature()
    metric = EuclideanMetric(feature)
    qb = QuickBundles(threshold=threshold, metric=metric)
    clusters = qb.cluster(trajectory_list)

    # -----------------基于角度信息(cos与sin)聚类---------------------
    # 上面通过quick_bundles划分中一大类,再根据方向划分为两大类,并用列表储存
    # 创建一个储存不同类型轨迹的列表
    track_dic = {}
    cls_track_dic = {}
    id_track_dic = {}
    # 记录轨迹类型,作为初始化类型
    k = 0
    # 角度等分数量(必须为2的倍数哈,最好是6的倍数) 角度为360/angle
    angle = 36
    # 遍历每一个类别
    for key, cls in enumerate(clusters):
        # 拿到同类别最长的一根线作为基准线
        # 遍历每一条轨迹得到轨迹的长度
        max_length = 0
        # 遍历每个类别中的每根线,为了得到最长的线
        for traj_id in cls.indices:
            # 得到每一条轨迹
            new_traj = trajectory_list[traj_id]
            new_length = cal_trajectory_length(new_traj)
            if new_length > max_length:
                max_length = new_length
                base_traj = new_traj
        # 基于最长的线作为base_vector
        start_pt = base_traj[0]
        end_pt = base_traj[-1]
        base_vector = end_pt - start_pt  # vector为numpy格式
        base_vector = normalize(base_vector)

        # 再遍历每一个轨迹id,计算向量
        for i, traj_id in enumerate(cls.indices):
            new_traj = trajectory_list[traj_id]  # 得到每一条轨迹
            new_cls = cls_trajectory_list[traj_id]  # 得到每一条轨迹的类型
            new_id = id_trajectory_list[traj_id]  # 得到每一条轨迹的车辆id
            start_pt = new_traj[0]
            end_pt = new_traj[-1]
            vector = end_pt - start_pt
            vector = normalize(vector)

            # 求向量余弦值与正弦值
            cos = sum(base_vector * vector)  # 余弦值
            cos = np.clip(cos, -1, 1)  # 将cos结果限制在-1~1之间
            # sin = sum(np.flip(base_vector) * vector)  # 正弦值
            sin = base_vector[1] * vector[0] - base_vector[0] * vector[1]
            # print(sin ** 2 + cos ** 2)
            sin = np.clip(sin, -1, 1)  # 将sin结果限制在-1~1之间
            # 计算相关性,与10度倍数的关系
            # 分情况分类,基于cos,sin函数的特点(注意sin划分线都不能搞到位于0,pi,2pi的位置)
            # 划分份数为n
            # 增加一个偏移项的取值为 (2*pi)/(2*n) = pi/n (以免在同方向微小差距会被分为不同类别)

            # 先通过cos判断方向正反
            # 正(0 < cos < 1)
            if 0 <= cos:
                # 再判断sin值(注意要先偏移一个小角度，因为我们不希望有压着边线的情况，不想压着横平竖直的方位进行划分)
                for i in range(angle // 2):
                    if math.sin(-pi / 2 + (2 * pi / angle) * i) <= math.sin(math.asin(sin) + pi / angle) <= math.sin(
                            -pi / 2 + (2 * pi / angle) * (i + 1)):
                        # 如果该类不在字典中,则需要新建一个列表
                        if k + i not in track_dic:
                            track_dic[k + i] = [new_traj]
                            cls_track_dic[k + i] = [new_cls]
                            id_track_dic[k + i] = [new_id]
                        # 如果该类别已存在,那么直接
                        else:
                            track_dic[k + i].append(new_traj)
                            cls_track_dic[k + i].append(new_cls)
                            id_track_dic[k + i].append(new_id)
            # 反(-1 <= cos <= 0)
            else:
                # 再判断sin值(注意要先偏移一个角度，因为我们不希望有压着边线的情况，不想压着横平竖直的方位进行划分)
                for i in range(angle // 2):
                    if math.sin(-pi / 2 + (2 * pi / angle) * i) <= math.sin(math.asin(sin) + pi / angle) <= math.sin(
                            -pi / 2 + (2 * pi / angle) * (i + 1)):
                        # 如果该类不在字典中,则需要新建一个列表
                        if k + i + angle // 2 not in track_dic:
                            track_dic[k + i + angle // 2] = [new_traj]
                            cls_track_dic[k + i + angle // 2] = [new_cls]
                            id_track_dic[k + i + angle // 2] = [new_id]
                        # 如果该类别已存在,那么直接
                        else:
                            track_dic[k + i + angle // 2].append(new_traj)
                            cls_track_dic[k + i + angle // 2].append(new_cls)
                            id_track_dic[k + i + angle // 2].append(new_id)

        # 一个大类整体再分k个小类
        # 这个k是用作标签的，是个兼容性误差的东西，一定要防止标签重复的情况
        k += angle  # 一般来说,是将上面的一大类再根据方向划分为36大类(每10度一类,但实际上有些类有可能是空的),长度再分两类（后面才用到，这里先预留一下空间，好像根本不需要，后面会另外再分）,所以乘以2

    # -----------------基于长度信息(轨迹)聚类---------------------
    # 再基于长度聚类分为,每个类别分两类
    # 满足最大长度的一定范围
    tmp_dic = deepcopy(track_dic)
    tmp_cls_dic = deepcopy(cls_track_dic)
    tmp_id_dic = deepcopy(id_track_dic)
    # print("track_dic", track_dic)
    # print("cls_track_dic", cls_track_dic)
    track_dic = {}
    cls_track_dic = {}
    id_track_dic = {}
    # 初始化分类标签key值
    k = 0
    # 遍历每一个类别
    for key in tmp_dic:
        tracks_length_list = []
        # 遍历每一条轨迹得到长度信息
        for traj in tmp_dic[key]:
            # 计算每一条轨迹的长度
            length = cal_trajectory_length(traj)
            tracks_length_list.append(length)
        # 获取某一类别轨迹的最大长度
        max_length = max(tracks_length_list)
        # 索引值
        i = 0
        # 遍历每一条轨迹
        for traj, cls, id in zip(tmp_dic[key], tmp_cls_dic[key], tmp_id_dic[key]):
            # 长度判断（范围设置为0.1，这里面的计算都是归一化之后的长度，都是一些小数值）
            # 改为设置为最大长度的0.9倍吧
            if tracks_length_list[i] >= 0.9 * max_length:
                # 如果该类不在字典中,则需要新建一个列表
                if k not in track_dic:
                    track_dic[k] = [traj]
                    cls_track_dic[k] = [cls]
                    id_track_dic[k] = [id]
                # 如果该类别已存在,那么直接
                else:
                    track_dic[k].append(traj)
                    cls_track_dic[k].append(cls)
                    id_track_dic[k].append(id)
            else:
                # 如果该类不在字典中,则需要新建一个列表
                if k + 1 not in track_dic:
                    track_dic[k + 1] = [traj]
                    cls_track_dic[k + 1] = [cls]
                    id_track_dic[k + 1] = [id]
                # 如果该类别已存在,那么直接
                else:
                    track_dic[k + 1].append(traj)
                    cls_track_dic[k + 1].append(cls)
                    id_track_dic[k + 1].append(id)
            i += 1  # 用于获取长度的索引值
        k += 2  # 每个小类分了两类

    # print(track_dic)
    # print(cls_track_dic)
    # print(len(track_dic))

    # ------最终数据整理---------
    # track_dic为最新分好的类
    # cls_track_dic为最新分好的类的对应车辆分类依据
    # 给不同类别一个代表性的曲线, 并计算数量, 数量作为粗细的依据
    # 基于轨迹检测长度最长的,作为代表性曲线

    track_count = []  #
    track_cls_dic = {}  # 记录每个轨迹类别的各类车数量
    # 遍历每一个类轨迹,计算每一个轨迹各类别车辆的数量
    for key in track_dic:
        bus = cls_track_dic[key].count(0)
        car = cls_track_dic[key].count(1)
        heavy_truck = cls_track_dic[key].count(2)
        medium_truck = cls_track_dic[key].count(3)
        midget_truck = cls_track_dic[key].count(4)
        # bus:0,car:1,heavy_truck:2,medium_truck:3,midget_truck:4
        # 记录每一个轨迹各类别车辆的数量
        track_cls_dic[key] = [bus, car, heavy_truck, medium_truck, midget_truck]
        # 获取每条轨迹的车辆总数
        track_count.append(len(track_dic[key]))

    # ------------筛选掉不同类轨迹数量最少的轨迹(后期筛选,于执行算法后)------------
    # 对轨迹总数求和
    # sum_count = sum(track_count)
    # 计算每个轨迹总数的百分比(列表形式)
    # track_count_percentage = np.array(track_count) / sum_count
    del_keys = []
    for i, key in enumerate(track_cls_dic):
        # 过滤掉百分比小于1%的线
        # if track_count_percentage[i] < 0.01:
        # 默认过滤掉数量不满足5辆的车辆
        if track_count[i] < min_cars:
            del_keys.append(key)

    # 根据key删除不要的轨迹
    for key in del_keys:
        del track_dic[key]
        del track_cls_dic[key]
        del id_track_dic[key]

    # 重命名字典的key（每次删完轨迹重新让key从0按顺序开始的逻辑）
    tmp_dic = {}
    tmp_cls_dic = {}
    tmp_id_dic = {}
    for i, key in enumerate(track_dic):
        tmp_dic[i] = track_dic[key]
        tmp_cls_dic[i] = track_cls_dic[key]
        tmp_id_dic[i] = id_track_dic[key]

    # 根据轨迹的总数对字典顺序重新排序
    # 各轨迹总数与各轨迹总数百分比计算
    track_count = []
    for key in track_cls_dic:
        track_count.append(sum(track_cls_dic[key]))
    sum_count = sum(track_count)
    track_count_percentage = np.array(track_count) / sum_count
    # 百分比排序后结果
    rank_key = np.argsort(-track_count_percentage)  # 排序index(大到小)
    track_count_percentage = np.flip(np.sort(track_count_percentage))  # 排序最终结果(大到小)

    # 得到排序后的字典(最终确定，根据排序重新让key从0按顺序开始的逻辑)
    track_dic = {}
    track_cls_dic = {}
    id_track_dic = {}
    for i, key in enumerate(rank_key):
        track_dic[i] = tmp_dic[key]
        track_cls_dic[i] = tmp_cls_dic[key]
        id_track_dic[i] = tmp_id_dic[key]

    # 用处于中位数的线作为代表线(用于显示)
    track_representation = []
    for key in track_dic:
        # 遍历每一条轨迹得到轨迹的长度
        tracks_length_list = []
        for new_traj in track_dic[key]:
            # 取每个类别中每个轨迹的长度
            each_length = cal_trajectory_length(new_traj)
            tracks_length_list.append(each_length)
        # 基于轨迹长度排序,然后得到轨迹最长的index(相对于原类别顺序)
        tmp = sorted(tracks_length_list)  # 排序
        middle_item = tmp[len(tracks_length_list) // 2]  # 排序后,取长度在中位数的长度
        middle_index = tracks_length_list.index(middle_item)  # 通过数值得到对应的index
        track_representation.append(track_dic[key][middle_index])

    # print(track_representation)
    # print(track_count)
    return track_dic, track_cls_dic, id_track_dic, track_representation, track_count_percentage


# 用cv2将聚类后的轨迹画在图上
def draw_lines(img_base, txt_path, threshold=0.125, min_cars=5):
    # 读取原始图像
    h = img_base.shape[0]
    w = img_base.shape[1]
    # 创建一个空白图像(全黑色)
    img = np.zeros((h, w, 3), np.uint8)
    # 转化为灰度图像(底图)
    imGray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    img_base = np.stack([imGray, imGray, imGray], axis=2)
    # 这里注意track_dic, track_representation结果是归一化的
    track_dic, track_cls_dic, id_track_dic, track_representation, track_count_percentage = cluster_tracks(txt_path, h,
                                                                                                          w,
                                                                                                          threshold=threshold,
                                                                                                          min_cars=min_cars)
    if len(track_representation) == 0:
        return 0
    # 将轨迹还原到原图的尺寸
    track_representation *= np.array([w, h])
    # 四舍五入并转化为像素点位置int类型
    track_representation = np.round(track_representation).astype(np.int32)
    # print('各轨迹各车辆类别数:', track_cls_dic)
    # 显示限制,只展示top8
    # track_representation = track_representation[:8]
    # track_count_percentage = track_count_percentage[:8]

    # 绘制颜色初始化(最多8个) RGB格式
    # colors = [[176, 58, 46],  # 1
    #           [160, 64, 0],  # 2
    #           [183, 149, 11],  # 3
    #           [118, 68, 138],  # 4
    #           [177, 48, 124],  # 5
    #           [40, 116, 166],  # 6
    #           [20, 143, 119],  # 7
    #           [30, 132, 73]]  # 8

    # 画框设置不同的颜色
    hsv_tuples = [(x / len(track_representation), 0.7, 0.9) for x in range(len(track_representation))]
    rgb_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    rgb_colors = list(map(lambda x: [int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)], rgb_colors))
    bgr_colors = []
    front_colors = []
    # 转为BGR格式
    for color in rgb_colors:
        bgr_colors.append(list(reversed(color)))
        # rgb在前端显示的数据处理, 处理成以下格式
        # ['rgb(176,58,46)', 'rgb(160,64,0)',
        #  'rgb(183,149,11)', 'rgb(118,68,138)',
        #  'rgb(177,48,124)', 'rgb(40,116,166)',
        #  'rgb(20,143,119)', 'rgb(30,132,73)']
        front_colors.append(f'rgb({str(color).replace("[", "").replace("]", "").replace(" ", "")})')

    # 绘图与箭头
    for i, r in enumerate(track_representation):
        p = track_count_percentage[i]
        color = bgr_colors[i]
        img = cv2.polylines(img, [r], isClosed=False, color=color, thickness=round(p * 200))
        img = cv2.arrowedLine(img, r[-2], r[-1], color=color, thickness=round(p * 200), line_type=cv2.LINE_8,
                              tipLength=math.sqrt(p) * 2)
    # img = 0.8 * img_base + img - np.array([30, 30, 30])
    img = 0.1 * img_base + img
    tmp_frame = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # iio.imwrite(txt_path.split('.')[0] + '.jpg', tmp_frame)
    # iio.imwrite(txt_path.replace('.txt', f'_{str(threshold).replace(".", "")}_{min_cars}.jpg'), tmp_frame)
    iio.imwrite(txt_path.replace('.txt', f'_result.jpg'), tmp_frame)

    # 作为后端输出的数据,列表套列表的格式
    track_count = []
    track_cls_count = []
    for key in track_cls_dic:
        track_count.append(sum(track_cls_dic[key]))
        track_cls_count.append(track_cls_dic[key])
        # # 在第8个结束循环
        # if key == 8:
        #     break
    count_result = [track_count] + track_cls_count
    print(count_result)
    print(front_colors)
    return count_result, front_colors


def visualize_tracks(track_dic):
    # QB算法结果可视化()以点的形式
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(track_dic))]
    # 遍历每一个类型
    for k, col in zip(track_dic, colors):
        points = []
        # 遍历每一条轨迹
        for traj in track_dic[k]:
            # 遍历轨迹中的每一个点
            for pt in traj:
                # 将同类的轨迹放在一个points里
                points.append(list(pt))
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=1)
    # 设置坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # 反转y轴方向
    plt.gca().invert_yaxis()
    plt.show()


# 生成过滤后，并拥有轨迹类别的txt文件
def output_result_txt(txt_path, id_track_dic, start_vector_list, end_vector_list):
    new_lines = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # 遍历txt每一行数据
    for line in lines:
        info_list = line.replace('\n', '').split(',')
        frame, id, x1, y1, x2, y2, conf, cls = info_list[0:8]

        # 新建一个列表用于存储过滤后的数据
        for key in id_track_dic:
            if id in id_track_dic[key]:
                new_lines.append([frame, id, x1, y1, x2, y2, conf, cls, str(key),
                                  str(start_vector_list[int(key)]), str(end_vector_list[int(key)])])

    output_txt = txt_path.replace('.txt', '_result.txt')
    with open(output_txt, 'w') as f:
        for sublist in new_lines:
            line = '/'.join(sublist)
            f.write(line + '\n')


# 获取代表轨迹的，起点方向向量，后面需要作为车行驶方向的判断开始
def get_track_representation_vector(track_representation):
    track_representation_start_vector = []
    track_representation_end_vector = []
    # 可使用类别
    # 东向西，北向南，西向东，南向北 东西——0, 2， 南北——1, -1
    start_direc = [['', '北向南', '南向北'], ['西向东', '西北向东南', '西南向东北'],
                   ['东向西', '东北向西南', '东南向西北']]
    # 左转，直行，右转
    direction_cls = []  # 比如：['东向西直行','北向南直行','西向东直行',...]

    for each_track in track_representation:
        # 获取起点向量与终点向量
        pt1 = each_track[0]
        pt2 = each_track[1]
        startVector = pt2 - pt1
        startVector = normalize(startVector)
        track_representation_start_vector.append(startVector.tolist())

        pt1 = each_track[-2]
        pt2 = each_track[-1]
        endVector = pt2 - pt1
        endVector = normalize(endVector)
        track_representation_end_vector.append(endVector.tolist())

        # 将起点向量转为整数判断起点方向
        # 定义分类的阈值，如小于0.3给0大于0.3给1
        # 角度按方向等分为八份
        seg_num = 8
        # 每一份对应的角度
        theta = 2 * pi / seg_num
        thre = math.cos(1.5 * theta)
        vector = []
        for val in startVector:
            if abs(val) < thre:
                vector.append(0)  # --x   0         1         -1
            elif val > 0:  # y
                vector.append(1)  # 0     /       西向东      东向西
            elif val < 0:  # 1   北向南   西北向东南   东北向西南
                vector.append(-1)  # -1  南向北   西南向东北   东南向西北
            else:
                raise ValueError('未定义的向量值')

        # print(startVector, vector)
        direction = start_direc[vector[0]][vector[1]]

        # 通过起点向量与终点向量的正弦值与余弦值判断轨迹转向类型
        vector_sin = endVector[1] * startVector[0] - endVector[0] * startVector[1]
        vector_cos = sum(startVector * endVector)  # 余弦值
        round_sin = round(vector_sin)
        # 对正弦值：左转为-1，右转为1，直行或掉头为0
        if round_sin == 0:
            # 直行
            if round(vector_cos) == 1:
                swerve = '直行'
            # 掉头
            elif round(vector_cos) == -1:
                swerve = '掉头'
            # 异常情况
            else:
                raise ValueError('未定义的cos值')
        # 右转
        elif round_sin == 1:
            swerve = '右转'
        # 左转
        elif round_sin == -1:
            swerve = '左转'
        # 异常情况
        else:
            raise ValueError('未定义的sin值')
        direction_cls.append(direction + '-' + swerve)
    # print(direction_cls)

    return track_representation_start_vector, track_representation_end_vector, direction_cls


if __name__ == '__main__':
    # 读取的txt数据
    txt_path = r'example\1.txt'
    # 底图图片
    image_path = r'example\1.jpg'
    # 超参
    threshold = 0.125
    min_cars = 5

    # 读取图片
    img_pil = Image.open(image_path)
    img_cv2 = np.array(img_pil)
    img_base = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    h, w = img_base.shape[:2]

    # 1.matplotlib可视化测试（用于测试检查）
    track_dic, track_cls_dic, id_track_dic, track_representation = cluster_tracks(txt_path, h, w)[0:4]
    visualize_tracks(track_dic)
    start_vector_list, end_vector_list, direction_cls = get_track_representation_vector(track_representation)
    print(start_vector_list)
    print(end_vector_list)
    print(direction_cls)
    output_result_txt(txt_path, id_track_dic, start_vector_list, end_vector_list)

    # 2.cv2绘图测试（实际用于可视化绘图的）
    count_result, front_colors = draw_lines(img_base, txt_path, threshold=threshold, min_cars=min_cars)
    print('轨迹总数:', sum(count_result[0]))

# # QB算法结果可视化
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(clusters))]
# for k, col in zip(clusters, colors):
#     points = []
#     # 遍历每一个
#     for traj_id in k.indices:
#         # 获取每一个id的序号
#         traj = trajectory_list[traj_id]
#         for pt in list(traj):
#             points.append(list(pt))
#     points = np.array(points)
#     plt.plot(points[:, 0], points[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=1)
# plt.show()

# print(len(trajectory))
# db = DBSCAN(eps=0.008, min_samples=55).fit(pts_array)
# labels = db.labels_
# unique_labels = set(labels)
# print(unique_labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# print(colors)
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#     class_member_mask = (labels == k)
#     xy = pts_array[class_member_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor=tuple(col), markersize=1)
# plt.title('Estimated number of clusters: %d' % len(unique_labels))
# plt.show()
# for id in trajectory:
#     traj = trajectory[id]
#     x = [point[0] for point in traj]
#     y = [point[1] for point in traj]
#     # 用三次多项式拟合点
#     z = np.polyfit(x, y, 3)
#     p = np.poly1d(z)
#     # 通过拟合后的曲线得到y
#     y_pre = p(x)
#     plt.plot(x, y_pre)
# plt.show()
