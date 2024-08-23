"""
1.车头时距的基本概念是指在同一车道上行驶的车辆队列中，前后两辆车的前端通过同一地点的时间差。
2.车头间距，又称为空间车头间距，是指同一车道上行驶的车辆之间，前车车尾与后车车头之间的实际距离。
3.排队长度指路口进口道各转向的排队长度；定义为从路口信号灯转为绿灯时刻，该路口进口道各转向车流排队最后一辆车距离路口停车线的距离。
4.速度，车辆通过有信号灯控制路口时的行车速度。
视频默认30fps，通过frame_id可以计算得到视频的时长
"""

import math
import numpy as np
import cv2
from PIL import Image
from track.trajectory_clustering import draw_lines
from matplotlib.path import Path


# 计算两点之间的距离
def calculate_distance(point1, point2):
    """
    计算两个像素点之间的欧几里得距离。

    参数:
    point1 (tuple): 第一个点的坐标，格式为(x1, y1)。
    point2 (tuple): 第二个点的坐标，格式为(x2, y2)。

    返回:
    float: 两点之间的距离。
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# 通过多边形得到掩码矩阵
def get_mask(h, w, mask_pt: list):
    # 创建图像
    img = np.zeros((h, w), np.uint8)
    # 遍历每一根多段线
    for pl in mask_pt:
        pl = np.array(pl)
        pl[:, 0] = np.round(pl[:, 0] * w)  # x
        pl[:, 1] = np.round(pl[:, 1] * h)  # y

        # 绘制多边形
        cv2.polylines(img, [np.array(pl, dtype=np.int32)], True, 1)
        # 获取掩码
        img = cv2.fillPoly(img, [np.array(pl, dtype=np.int32)], 1)

    # cv2.imshow('Mask Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


# 通过多边形得到掩码矩阵与mask中点
def get_each_mask(h, w, mask_pt: list):
    img_list = []
    # 遍历每一根多段线
    for pl in mask_pt:
        # 创建图像
        img = np.zeros((h, w), np.uint8)
        pl = np.array(pl)
        pl[:, 0] = np.round(pl[:, 0] * w)  # x
        pl[:, 1] = np.round(pl[:, 1] * h)  # y

        # 绘制多边形
        cv2.polylines(img, [np.array(pl, dtype=np.int32)], True, 1)
        # 获取掩码
        img = cv2.fillPoly(img, [np.array(pl, dtype=np.int32)], 1)
        img_list.append(img)

    # cv2.imshow('Mask Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_list


# 停止线数据去归一化
def unormalize_line(h, w, stop_line):
    new_line = [[[int(x * w), int(y * h)] for x, y in line] for line in stop_line]
    return new_line


# 返回出口道mask对应的方向类型
def get_exit_direct(h, w, mask_pt: list):
    # 八个区域对应的方向信息
    direct_list = [[['东南向西北-直行', '西南向东北-左转', '东北向西南-右转', '西北向东南-掉头'],
                   ['南向北-直行', '西向东-左转', '东向西-右转', '北向南-掉头'],
                   ['西南向东北-直行', '西北向东南-左转', '东南向西北-右转', '东北向西南-掉头']],
                   [['东向西-直行', '南向北-左转', '北向南-右转', '西向东-掉头'],
                    [],
                   ['西向东-直行', '北向南-左转', '南向北-右转', '东向西-掉头']],
                   [['东北向西南-直行', '东南向西北-左转', '西北向东南-右转', '西南向东北-掉头'],
                   ['北向南-直行', '东向西-左转', '西向东-右转', '南向北-掉头'],
                   ['西北向东南-直行', '东北向西南-左转', '西南向东北-右转', '东南向西北-掉头']]]
    # print(direct_list[0][0], direct_list[1][1])
    midpoint_list = []
    for pl in mask_pt:
        # 提取所有顶点的坐标
        x_coords = [point[0] for point in pl]
        y_coords = [point[1] for point in pl]

        # 计算中点
        x = sum(x_coords) / len(x_coords)
        y = sum(y_coords) / len(y_coords)
        # print(x, y)
        midpoint = []
        for pt in [x, y]:
            if pt < 1/3:
                midpoint.append(0)
            elif pt < 2/3:
                midpoint.append(1)
            else:
                midpoint.append(2)
    #     print(midpoint)
    #
    # print(midpoint_list)


# 计算检测框的中点（作为车辆的位置）
def calculate_midpoint(info_list):
    x_center = (float(info_list['x1']) + float(info_list['x2'])) / 2
    y_center = (float(info_list['y1']) + float(info_list['y2'])) / 2
    return [int(x_center), int(y_center)]


# 辅助函数：判断点是否在mask内
def is_point_in_mask(point, mask):
    # 实现点在多边形内的判断逻辑
    x1, y1 = point
    result = mask[y1, x1]
    # print(result)
    return result


def reset_index(index, index_max):
    """重置索引到0，如果索引达到最大值"""
    return (index + 1) % index_max


# 筛选清洗数据异常值——出现分错类别的轨迹
def delete_cls(cars_dict, min_cars):
    # 步骤1: 统计每个 track_cls 出现的次数
    track_cls_count = {}
    for entry in cars_dict.values():
        track_cls = entry['track_cls']
        if track_cls in track_cls_count:
            track_cls_count[track_cls] += 1
        else:
            track_cls_count[track_cls] = 1
    # 步骤2: 筛选出现次数小于5的 track_cls 值
    track_cls_to_remove = [cls for cls, count in track_cls_count.items() if count < min_cars]
    # 步骤3: 删除对应的数据
    keys_to_remove = [key for key, value in cars_dict.items() if value['track_cls'] in track_cls_to_remove]
    for key in keys_to_remove:
        del cars_dict[key]
    return cars_dict


# 计算某一方向车头时距
def calculate_headway_time(lanes_arrays):
    # 默认帧率
    fps = 30
    sum_time = 0
    # 计算总平均车头时距的分母，一般为车道数，特殊情况会进行删减，如有车道对应排到的车只有一辆不构成时间的计算
    lanes_lens = len(lanes_arrays)
    for each_lane in lanes_arrays:
        # 若单车道对应的车只有一辆，则跳过计算
        if len(each_lane) == 1:
            lanes_lens -= 1
            continue
        # 计算单个车道的平均时距
        each_time = (each_lane[-1] - each_lane[0]) / (len(each_lane) - 1) / fps
        sum_time += each_time
    # 计算车头时距总的平均值
    # 假如车道数与车辆数相等，即刚好每条车道分到都只有一辆，无法计算，返回None
    if lanes_lens == 0:
        return None
    else:
        average_time = round(sum_time / lanes_lens, 2)
        return average_time


# 计算单一一方向车头间距
def calculate_headway_distance(lanes_arrays, length_per_pixel):
    sum_distance = 0
    lanes_lens = len(lanes_arrays)
    for each_lane in lanes_arrays:
        # 单个车道对应的车头间距
        each_distance = 0
        if len(each_lane) == 1:
            lanes_lens -= 1
            continue
        # 计算单个车道的平均车头间距
        for i in range(len(each_lane) - 1):
            # 遍历单个车道的车辆数据并进行累加
            each_distance += calculate_distance(each_lane[i], each_lane[i + 1]) * length_per_pixel
            # print(calculate_distance(each_lane[i], each_lane[i + 1]) * length_per_pixel)
        # 计算单个车道的平均值
        each_distance = each_distance / (len(each_lane) - 1)
        # 总间距的累加
        sum_distance += each_distance
    if lanes_lens == 0:
        return None
    else:
        # 计算平均车头间距
        sum_distance = sum_distance / lanes_lens
        # print(sum_distance)
        return sum_distance


# 计算点到线段的距离
def calculate_pt_to_segment(pt, segment):
    # 先转为np格式
    pt = np.array(pt)
    segment_A = np.array(segment[0])
    segment_B = np.array(segment[1])

    # 步骤1：计算线段向量，线段一端点到点的向量
    AP = pt - segment_A
    AB = segment_B - segment_A

    # 步骤2：计算向量AP在AB方向上的投影向量
    proj = np.dot(AP, AB) * AB / np.dot(AB, AB)
    # 步骤3：计算投影点P'
    p_proj = segment_A + proj
    # 步骤4：计算点P到投影点P'的距离（即点到线段的距离）
    pt_distance = calculate_distance(pt, p_proj)

    return pt_distance


# 通过计算mask区域到停止线的距离返回对应停止线
def calculate_mask_to_line(entrance_areas, stop_segments, h, w):
    # 存储每个mask对应的停止线
    area_lines = [[] for _ in range(len(entrance_areas))]
    # 存储每个mask的中点
    mask_midpoints = []
    for each_area in entrance_areas:
        x_coords = [point[0] * w for point in each_area]
        y_coords = [point[1] * h for point in each_area]

        # 计算当前闭合区域的中点
        x = sum(x_coords) / len(x_coords)
        y = sum(y_coords) / len(y_coords)
        mask_midpoints.append([x, y])
    # print(mask_midpoints)

    # 分别计算中点到停止线的距离
    for segment in stop_segments:
        # 最小的距离（先给定一个大值）
        min_distance = w
        # 最小距离对应的index
        min_index = 0
        # print('----------------------------------------------------------------------------------')
        # print(segment)
        # 停止线的中点
        stop_midpoint = [(segment[1][0] + segment[0][0]) / 2, (segment[1][1] + segment[0][1]) / 2]
        for i, mask_midpoint in enumerate(mask_midpoints):
            pt_distance = calculate_distance(stop_midpoint, mask_midpoint)
            # 判断停止线到mask区域的距离，并获取最短的距离与对应的索引
            if pt_distance < min_distance:
                min_distance = pt_distance
                min_index = i
            # print(pt_distance)
        area_lines[min_index].append(segment)
        # print(min_distance, min_index)

    # 调整顺序，若有右转专用道，保持先直行停止线后右转停止线的顺序
    for i, each_line in enumerate(area_lines):
        # 同一进口道区域停止线不止一条时
        if len(each_line) > 1:
            length0 = calculate_distance(each_line[0][0], each_line[0][1])
            length1 = calculate_distance(each_line[1][0], each_line[1][1])

            # 调整内部顺序，长的为直行，放前面，短的为右转，放后面
            if length0 > length1:
                straight = each_line[0]
                right = each_line[1]
            else:
                # 如果第二条线段更长，交换它们的位置
                straight = each_line[1]
                right = each_line[0]
            area_lines[i][0] = straight
            area_lines[i][1] = right

    return area_lines


def is_car_stop(pt1, pt2, x_tolerance=0.001, y_tolerance=0.0015):
    # 车辆当前数据与存储的上一次的数据对比，判断两次中点是否在阈值内趋于重合
    if abs(pt1[0] - pt2[0]) < x_tolerance and abs(pt1[1] - pt2[1]) < y_tolerance:
        # 两点距离小于阈值则判断为重合
        return True
    else:
        return False


def get_2_stage_cars(further_car_dict, stop_car_dict, lanes_num, mid_point,
                     pt_to_line_distance, each_car_info, h, w):
    # TODO:有没有更长的（对比dict2）——有没有出现过（dict1）——是不是静止（对比dict1）——替换dict2——用dict2判断有没有行驶
    #   步骤1：计算距离最长的n辆车，若出现新的——加入dict1（dict1不限长度）
    #   步骤2：判断这辆车是否静止，用dict1中出现过的id进行判断，并将该id的新值替换进dict1
    #   （修改is_car_move函数，把针对一条道的修改为针对一辆车的识别）
    #   步骤3：静止则加入待计算列表中（dict2长度为n，按照原逻辑有新的更长的就替换）
    #   步骤4：（最后的）用最长的n辆车计算
    # print(further_car_dict)
    # print(each_car_info['id'], mid_point, pt_to_line_distance)
    # 填入与车道数相同数量的距离最近的车，存储信息包括在画面中的位置（中点）和到停止线的距离
    # 定义布尔值存储该车是否比原有车辆距离更长
    is_further = False
    for car_id, distance in stop_car_dict.items():
        # 存储车的数量与车道数相等时，判断当前数据是否为更长的
        if pt_to_line_distance > distance and each_car_info['id'] not in stop_car_dict:
            is_further = True
            break
    # 存储最远停止车辆的变量还没满（没到车道数），或满了但比原有距离长时
    if len(stop_car_dict) < lanes_num or is_further:
        # print(each_car_info['frame'], further_car_dict.keys(), each_car_info['id'])
        # 对出现过的车判断是不是静止
        if each_car_info['id'] in further_car_dict:
            # # 判断车辆是否为停止状态
            # if is_car_stop():
            if is_car_stop(mid_point, further_car_dict[each_car_info['id']][0],
                           x_tolerance=0.002 * w, y_tolerance=0.003 * h):
                # 有距离更远的车辆处于静止状态，则将其到停止线的距离添加进存储停止车辆的列表
                stop_car_dict[each_car_info['id']] = pt_to_line_distance
                if len(stop_car_dict) > lanes_num:
                    # 使用 min 函数找到最小 value 对应的 key，加入距离更大的值后，总数超出车道数量时删除这个最小的
                    min_key = min(stop_car_dict, key=lambda k: stop_car_dict[k])
                    del stop_car_dict[min_key]
        # 没有出现过则添加进字典中，出现过则覆盖，保证further_car_dict中是最新的信息（需要对比的旧数据前面用完不需要了就覆盖掉）
        further_car_dict[each_car_info['id']] = [mid_point, pt_to_line_distance, each_car_info['frame']]


# 计算排队长度
def calculate_traj_queue_length(lanes_num, current_farthest_car, length_per_pixel):
    # 车辆数量还不够平均每车道一辆，数据没有计算意义
    if len(current_farthest_car) < lanes_num:
        return None
    else:
        # 总长度
        length_sum = 0
        for length in current_farthest_car.values():
            length_sum += length
        # 计算平均排队长度
        queue_length = length_sum / lanes_num * length_per_pixel

        # print(queue_length)
        return queue_length


# 计算车头时距
# 车头时距的基本概念是指在同一车道上行驶的车辆队列中，”前后两辆车“的”前端“通过同一地点的时间差（使用出口道的停止线）。
def calculate_headway_times(info_list, length_per_pixel, exit_mask, exit_lane_num, min_cars):
    # 需要知道前一辆车的位置在哪
    # 基于汽车id来分
    car_list = []
    # 存储车辆时距顺序输出:{0: xx, 1: xx, ...}
    sequence_time_dic = {}
    # 遍历每一个mask
    for each_mask in exit_mask:
        car_dict = {}
        # 判断中心点是否位于mask区域内
        for track_info in info_list:
            center_x, center_y = calculate_midpoint(track_info)
            if is_point_in_mask((center_x, center_y), each_mask):
                # 基于轨迹类型对获取每个id的在路口区域的轨迹
                if track_info['id'] not in car_dict:
                    car_dict[track_info['id']] = track_info
        car_list.append(car_dict)

    # 遍历四个mask
    for i, each_car in enumerate(car_list):
        if len(each_car) != 0:
            # 删掉分错类别的轨迹（按照筛选轨迹的最小数量来分）
            each_car = delete_cls(each_car, min_cars)
            # 存储所有类别，按照左转、直行、右转的顺序
            all_cls = [[] for _ in range(3)]
            # 索引值初始化
            index = [0, 0, 0]
            # 索引最大值（车道数量）
            # 每个方向车道的最大可行驶的车道数
            # 判断是否有右转专用道，若没有右转专用道（即表示右转的车道数为0），右转转入的车道数与左转与直行相同
            if exit_lane_num[i][1] == 0:
                passable_lanes_num_list = [exit_lane_num[i][0], exit_lane_num[i][0], exit_lane_num[i][0]]
            else:
                passable_lanes_num_list = [exit_lane_num[i][0], exit_lane_num[i][0], exit_lane_num[i][1]]
            # 生成二维空数组的列表,通过出口道最大（这里相当于打算用一个小技巧，按顺序依次插入到每个列表中，是一种我们自己假设的理想情况）
            lanes_arrays = [[[] for _ in range(passable_lanes_num_list[0])], [[] for _ in range(passable_lanes_num_list[1])],
                            [[] for _ in range(passable_lanes_num_list[2])]]
            # print('---------------------------------------------------------')
            # 遍历每一辆车首次出现在出口道mask内的信息
            for j, each_frame in enumerate(each_car.values()):
                # print(car['track_cls'], car['direction_cls'])
                # 将行驶到同一个mask的区域按照不同转向方向进行划分，因为它们不会同时出现。然后将每个方向第一帧记录下来
                # 直行
                if '直行' in each_frame['direction_cls']:
                    lanes_arrays[1][index[1]].append(each_frame['frame'])
                    # 将直行的轨迹类别track_cls加入到基于direction_cls创建的列表，这种主要是考虑到有多条轨迹的情况
                    if each_frame['track_cls'] not in all_cls[1]:
                        all_cls[1].append(each_frame['track_cls'])
                    # 索引递增
                    # 索引递增和重置--数值到passable_lanes_num，超过最大车道数时，重新回到0塞入对应的列表中
                    index[1] = reset_index(index[1], passable_lanes_num_list[1])
                elif '右转' in each_frame['direction_cls']:
                    lanes_arrays[2][index[2]].append(each_frame['frame'])
                    if each_frame['track_cls'] not in all_cls[2]:
                        all_cls[2].append(each_frame['track_cls'])
                    index[2] = reset_index(index[2], passable_lanes_num_list[2])
                # 左转和掉头
                else:
                    lanes_arrays[0][index[0]].append(each_frame['frame'])
                    if each_frame['track_cls'] not in all_cls[0]:
                        all_cls[0].append(each_frame['track_cls'])
                    index[0] = reset_index(index[0], passable_lanes_num_list[0])

            # 计算车头时距
            for k, cls_list in enumerate(all_cls):
                # 先判断是否有数据
                if len(cls_list) != 0:
                    average_time = calculate_headway_time(lanes_arrays[k])
                    # print(average_time, cls_list)
                    for cls in cls_list:
                        if cls not in sequence_time_dic:
                            sequence_time_dic[cls] = average_time
    # print(sequence_time_dic)

    sequence_time_list = []
    # 排序轨迹
    for i in range(len(sequence_time_dic)):
        sequence_time_list.append(sequence_time_dic[i])
    print(sequence_time_list)

    return sequence_time_list


# 车头间距
# 车头间距，又称为空间车头间距，是指同一车道上行驶的车辆之间（进入出口道），”前车车头“与”后车车头“之间的实际距离。
def calculate_headway_distances(info_list, length_per_pixel, exit_mask, exit_lane_num, min_cars):
    # 需要知道前一辆车的位置在哪
    # 基于汽车id来分
    car_list = []
    # 存储平均车头间距顺序输出:{0: xx, 1: xx, ...}
    sequence_distance_dic = {}
    # 遍历每一个mask
    for each_mask in exit_mask:
        car_dict = {}
        # 判断中心点是否位于mask区域内
        for track_info in info_list:
            center_x, center_y = calculate_midpoint(track_info)
            if is_point_in_mask((center_x, center_y), each_mask):
                # 基于轨迹类型对获取每个id的在路口区域的轨迹（取刚进入区域的第一帧）
                if track_info['id'] not in car_dict:
                    car_dict[track_info['id']] = track_info
        car_list.append(car_dict)

    # 遍历四个mask(逻辑与计算车头时距差不多)
    for i, each_car in enumerate(car_list):
        if len(each_car) != 0:
            each_car = delete_cls(each_car, min_cars)
            # 存储所有类别，按照左转、直行、右转的顺序
            all_cls = [[] for _ in range(3)]
            index = [0, 0, 0]
            if exit_lane_num[i][1] == 0:
                passable_lanes_num_list = [exit_lane_num[i][0], exit_lane_num[i][0], exit_lane_num[i][0]]
            else:
                passable_lanes_num_list = [exit_lane_num[i][0], exit_lane_num[i][0], exit_lane_num[i][1]]
            lanes_arrays = [[[] for _ in range(passable_lanes_num_list[0])],
                            [[] for _ in range(passable_lanes_num_list[1])],
                            [[] for _ in range(passable_lanes_num_list[2])]]
            for j, each_info in enumerate(each_car.values()):
                # 计算该条车辆识别框的中点（作为计算距离的数据）
                mid_point = calculate_midpoint(each_info)
                if '直行' in each_info['direction_cls']:
                    # 将车辆位置（用识别框中点表示）依次存入表示车道的列表
                    lanes_arrays[1][index[1]].append(mid_point)
                    if each_info['track_cls'] not in all_cls[1]:
                        all_cls[1].append(each_info['track_cls'])
                    # 索引递增和重置
                    index[1] = reset_index(index[1], passable_lanes_num_list[1])
                elif '右转' in each_info['direction_cls']:
                    lanes_arrays[2][index[2]].append(mid_point)
                    if each_info['track_cls'] not in all_cls[2]:
                        all_cls[2].append(each_info['track_cls'])
                    index[2] = reset_index(index[2], passable_lanes_num_list[2])
                # 左转和掉头
                else:
                    lanes_arrays[0][index[0]].append(mid_point)
                    if each_info['track_cls'] not in all_cls[0]:
                        all_cls[0].append(each_info['track_cls'])
                    index[0] = reset_index(index[0], passable_lanes_num_list[0])
            # print(lanes_arrays)

            # 计算车头间距：利用计算前后车中点的距离近似作为两车车头的距离
            for k, cls_list in enumerate(all_cls):
                # 先判断是否有数据
                if len(cls_list) != 0:
                    average_distance = calculate_headway_distance(lanes_arrays[k], length_per_pixel)
                    for cls in cls_list:
                        if cls not in sequence_distance_dic:
                            sequence_distance_dic[cls] = average_distance
    # print(sequence_distance_dic)

    sequence_distance_list = []
    # 排序轨迹
    for i in range(len(sequence_distance_dic)):
        sequence_distance_list.append(sequence_distance_dic[i])
    print(sequence_distance_list)

    return sequence_distance_list


# 排队长度
# 排队长度指路口进口道各转向的排队长度；定义为从路口信号灯转为绿灯时刻，该路口进口道各转向车流排队最后一辆车距离路口停止线的距离。
# 直接算每根线的直线距离吧，然后选一根最短的
def calculate_queue_length(info_list, length_per_pixel, stop_segments, entrance_lane_num, h, w, entrance_areas):
    # TODO:需要判断车辆在什么情况处于排队状态
    # TODO:需要知道停止线的位置
    # 计算进口道对应的mask
    entrance_mask = get_each_mask(h, w, entrance_areas)
    # 先将车辆分到四个区域（进口道mask）
    car_list = []
    # 存储轨迹对应的排队长度
    queue_length_dict = {}
    # 遍历每一个mask
    for each_mask in entrance_mask:
        car_dict = {}
        # 判断中心点是否位于mask区域内
        for track_info in info_list:
            center_x, center_y = calculate_midpoint(track_info)
            if is_point_in_mask((center_x, center_y), each_mask):
                # 保存进口道区域的车辆信息（以帧数为索引逐帧保存）
                if track_info['frame'] not in car_dict:
                    car_dict[track_info['frame']] = [track_info]
                else:
                    car_dict[track_info['frame']].append(track_info)
        car_list.append(car_dict)

    # 判断不同进口道mask所对应的停止线，并用一个list按照mask的相同顺序存储
    # 计算返回每个mask对应的停止线
    area_lines = calculate_mask_to_line(entrance_areas, stop_segments, h, w)
    for i, each_area_cars in enumerate(car_list):
        if len(each_area_cars) != 0:
            # print('-----------------------------------------------------------------')
            # 每个方向对应的车道数，按照左转、直行、右转的顺序
            lanes_num_list = [entrance_lane_num[i][0] + entrance_lane_num[i][1],
                              entrance_lane_num[i][2] + entrance_lane_num[i][1] + entrance_lane_num[i][3],
                              entrance_lane_num[i][4] + entrance_lane_num[i][3]]
            # 存储所有类别，按照左转、直行、右转的顺序
            all_cls = [[] for _ in range(3)]
            # 每一帧的所有车
            # 存储最远的车，出现距离更远的车辆就写入，每获取到新一帧的数据，将已存在的车辆信息替换
            further_car_info = [{}, {}, {}]
            stop_car_info = [{}, {}, {}]
            for frame, each_frame_cars in each_area_cars.items():
                # 跳帧：先用每5秒取一帧
                jump_second = 3
                jump_num = jump_second * 30 / 2
                if frame % jump_num == 0:
                    # 最开始初始化一个空值，之后用的就是上一次算出来的停止车辆
                    last_stop_dict = stop_car_info
                    # 存储离停止线距离最远的车辆，数量与车道数相等，先判断是否静止再写入，每次读取新一帧重新初始化
                    stop_car_info = [{}, {}, {}]
                    # 当前帧下的每一辆车
                    for each_car in each_frame_cars:
                        # 计算检测框的中点
                        mid_point = calculate_midpoint(each_car)
                        if '直行' in each_car['direction_cls']:
                            # 计算到停止线的距离
                            pt_to_line_distance = calculate_pt_to_segment(mid_point, area_lines[i][0])
                            # 统计车辆轨迹类别
                            if each_car['track_cls'] not in all_cls[1]:
                                all_cls[1].append(each_car['track_cls'])
                            # 计算距离最短与距离最长的车
                            get_2_stage_cars(further_car_info[1], stop_car_info[1], lanes_num_list[1],
                                                 mid_point, pt_to_line_distance, each_car, h, w)

                        elif '右转' in each_car['direction_cls']:
                            # 有右转专用道时，计算距离用中点计算，不用点到线段的距离
                            if len(area_lines[i]) > 1:
                                line_x_center = (float(area_lines[i][1][0][0]) + float(area_lines[i][1][1][0])) / 2
                                line_y_center = (float(area_lines[i][1][0][1]) + float(area_lines[i][1][1][1])) / 2
                                pt_to_line_distance = calculate_distance(mid_point, [line_x_center, line_y_center])
                            # 无右转专用道时，停止线通用
                            else:
                                pt_to_line_distance = calculate_pt_to_segment(mid_point, area_lines[i][0])
                            # 统计车辆轨迹类别
                            if each_car['track_cls'] not in all_cls[2]:
                                all_cls[2].append(each_car['track_cls'])
                            # 计算距离最短与距离最长的车
                            get_2_stage_cars(further_car_info[2], stop_car_info[2], lanes_num_list[2],
                                                 mid_point, pt_to_line_distance, each_car, h, w)
                        # 左转和掉头
                        else:
                            pt_to_line_distance = calculate_pt_to_segment(mid_point, area_lines[i][0])
                            # 统计车辆轨迹类别
                            if each_car['track_cls'] not in all_cls[0]:
                                all_cls[0].append(each_car['track_cls'])
                            # 计算距离最短与距离最长的车
                            get_2_stage_cars(further_car_info[0], stop_car_info[0], lanes_num_list[0],
                                                 mid_point, pt_to_line_distance, each_car, h, w)
                    # print(last_stop_dict)
                    # print(stop_car_info)
                    # print(all_cls)
                    # print('--------------------------------------')

                    for j in range(len(stop_car_info)):
                        # 上一次有算出停止车辆，但当前停止车辆dict为空或小于对应车道数，则认为车辆由停止转为行驶，信号灯由红转绿
                        if len(last_stop_dict[j]) == lanes_num_list[j] and len(stop_car_info[j]) < lanes_num_list[j]:
                            # 计算车辆排队长度平均值
                            average_length = sum(last_stop_dict[j].values()) / lanes_num_list[j] * length_per_pixel
                            # 保存到类别对应的字典中
                            for cls in all_cls[j]:
                                if cls not in queue_length_dict:
                                    queue_length_dict[cls] = average_length

            # 当前mask所有计算结束后的数据补充，没有计算到对应值的轨迹返回None
            for k, cls_list in enumerate(all_cls):
                # 先判断是否有数据
                if len(cls_list) != 0:
                    for cls in cls_list:
                        if cls not in queue_length_dict:
                            queue_length_dict[cls] = None
    # print(queue_length_dict)

    queue_length_list = []
    # 排序轨迹
    for i in range(len(queue_length_dict)):
        queue_length_list.append(queue_length_dict[i])
    print(queue_length_list)

    pass


# 速度
# 速度可以通过计算车辆在连续两帧之间的移动距离除以时间差来计算。（用起点终点）
# 计算路口围合区域的平均速度
def calculate_speed_at_intersection(info_list, length_per_pixel, mask):
    # 默认帧率
    fps = 30
    # 初始化存储起点终点与速度的字典
    start_dic = {}
    end_dic = {}
    distance_dic = {}
    speed_dic = {}
    average_dic = {}

    for info in info_list:
        # 得到车辆所在位置对应的点
        mid_point = calculate_midpoint(info)
        # 先判断是否在十字路口围合区域内
        if mask[mid_point[1]][mid_point[0]] == 1:
            # 如果该id不在起点字典中，则为第一次出现，新建一个存储起点坐标
            if info['id'] not in start_dic:
                start_dic[info['id']] = [mid_point]
                # 存储时间
                start_dic[info['id']].append(info['frame'])
                # 存储类别
                start_dic[info['id']].append(info['track_cls'])
            # 如果该id已存在起点字典中，则计算终点坐标与行驶距离
            else:
                # print('----------------------------')
                # print(info['id'])
                if info['id'] not in distance_dic:
                    distance_dic[info['id']] = calculate_distance(start_dic[info['id']][0], mid_point)
                else:
                    # print(distance_dic[info['id']], calculate_distance(end_dic[info['id']][0], mid_point))
                    distance_dic[info['id']] += calculate_distance(end_dic[info['id']][0], mid_point)
                end_dic[info['id']] = [mid_point]
                end_dic[info['id']].append(info['frame'])
                # print(distance_dic[info['id']])
    # print(start_dic)
    # print(end_dic)

    # 判断起点与终点的个数是否相同，不相同则把出意外的点删掉
    if len(start_dic) != len(end_dic):
        # print('起点终点长度不相等：', len(start_dic), len(end_dic))
        del_list = []
        for car_id in start_dic:
            if car_id not in end_dic:
                del_list.append(car_id)
        for key in del_list:
            del start_dic[key]

    # 计算起点与终点的距离
    for car_id in start_dic:
        distance = distance_dic[car_id] * length_per_pixel
        # 取帧数计算时间
        time = (end_dic[car_id][1] - start_dic[car_id][1]) / fps
        speed = distance / time * 3.6
        # 如果类别信息是否已存储
        if start_dic[car_id][2] not in speed_dic:
            speed_dic[start_dic[car_id][2]] = [speed]
        else:
            speed_dic[start_dic[car_id][2]].append(speed)

    for key, values in speed_dic.items():
        # 计算每组数据的平均值
        average_value = sum(values) / len(values)
        # 将结果存储在新的字典中
        average_dic[key] = average_value
    # # 计算平均速度(m/s)
    # average_speed = sum(speed_list) / len(speed_list)
    print('average_speed', average_dic)

    return average_dic


# ---------------第一组测试数据---------------
# 读取的txt数据
txt_path = r'example\1.txt'
# 底图图片
image_path = r'example\1.jpg'
# 超参
threshold = 0.125
min_cars = 5

# 画面尺寸
# w, h = (1920, 1080)

mask = []  # 车道区域
scale_line = [[0.4934539794921875, 0.2621527777777778], [0.5608367919921875, 0.265625]]  # 比例尺线
scale_length = 4 * 3.5  # 比例尺的实际尺寸，以m为单位
entrance_areas = [[[0.4446258544921875, 0.2708333333333333], [0.4631805419921875, 0.2934027777777778],
                   [0.4905242919921875, 0.2604166666666667], [0.5618133544921875, 0.2621527777777778],
                   [0.5657196044921875, 0.001736111111111111], [0.4856414794921875, 0.001736111111111111],
                   [0.4622039794921875, 0.203125]],
                  [[0.0012664794921875, 0.5503472222222222], [0.4114227294921875, 0.5451388888888888],
                   [0.4163055419921875, 0.6927083333333334], [0.4006805419921875, 0.7239583333333334],
                   [0.3840789794921875, 0.7534722222222222], [0.3186492919921875, 0.7291666666666666],
                   [0.0950164794921875, 0.7413194444444444], [0.0032196044921875, 0.7361111111111112]],
                  [[0.5129852294921875, 0.8003472222222222], [0.5862274169921875, 0.8020833333333334],
                   [0.6086883544921875, 0.7986111111111112], [0.6233367919921875, 0.8315972222222222],
                   [0.6096649169921875, 0.8819444444444444], [0.6018524169921875, 0.953125],
                   [0.6038055419921875, 0.9930555555555556], [0.5120086669921875, 0.9930555555555556]],
                  [[0.6448211669921875, 0.3975694444444444], [0.6565399169921875, 0.3576388888888889],
                   [0.6721649169921875, 0.3333333333333333], [0.9993133544921875, 0.3611111111111111],
                   [0.9983367919921875, 0.5138888888888888], [0.9534149169921875, 0.5190972222222222],
                   [0.6516571044921875, 0.5243055555555556]]]  # 进口道区域
entrance_lane_num = [[2, 0, 2, 0, 1], [2, 0, 3, 0, 1], [2, 1, 1, 0, 1], [1, 0, 3, 0, 1]]
exit_areas = [[[0.4436492919921875, 0.2630208333333333], [0.4592742919921875, 0.2942708333333333],
               [0.4163055419921875, 0.3637152777777778], [0.4065399169921875, 0.3880208333333333],
               [0.4065399169921875, 0.5199652777777778], [0.0012664794921875, 0.5217013888888888],
               [0.0041961669921875, 0.3602430555555556], [0.3498992919921875, 0.3602430555555556],
               [0.3918914794921875, 0.3376736111111111], [0.4260711669921875, 0.2994791666666667]],
              [[0.5598602294921875, 0.2595486111111111], [0.5940399169921875, 0.2578125],
               [0.6116180419921875, 0.2873263888888889], [0.6418914794921875, 0.3480902777777778],
               [0.6575164794921875, 0.3619791666666667], [0.6692352294921875, 0.3324652777777778],
               [0.6526336669921875, 0.3098958333333333], [0.6301727294921875, 0.2647569444444444],
               [0.6086883544921875, 0.20572916666666666], [0.5969696044921875, 0.12760416666666666],
               [0.5911102294921875, 0.04600694444444445], [0.5911102294921875, -0.0008680555555555555],
               [0.5618133544921875, 0.0008680555555555555]],
              [[0.6409149169921875, 0.5512152777777778], [0.6399383544921875, 0.6727430555555556],
               [0.6760711669921875, 0.6779513888888888], [0.6790008544921875, 0.6935763888888888],
               [0.6428680419921875, 0.7352430555555556], [0.6233367919921875, 0.7595486111111112],
               [0.6086883544921875, 0.7960069444444444], [0.6252899169921875, 0.8289930555555556],
               [0.6506805419921875, 0.7769097222222222], [0.6770477294921875, 0.7352430555555556],
               [0.7336883544921875, 0.7126736111111112], [0.8176727294921875, 0.7126736111111112],
               [0.9963836669921875, 0.7039930555555556], [0.9954071044921875, 0.5442708333333334]],
              [[0.4006805419921875, 0.7248263888888888], [0.3811492919921875, 0.7560763888888888],
               [0.4163055419921875, 0.8064236111111112], [0.4358367919921875, 0.8324652777777778],
               [0.4456024169921875, 0.8862847222222222], [0.4543914794921875, 0.9626736111111112],
               [0.4543914794921875, 0.9921875], [0.5110321044921875, 0.9939236111111112],
               [0.5110321044921875, 0.8828125], [0.5129852294921875, 0.8237847222222222],
               [0.4758758544921875, 0.8203125], [0.4700164794921875, 0.8307291666666666],
               [0.4582977294921875, 0.8255208333333334], [0.4397430419921875, 0.7786458333333334],
               [0.4163055419921875, 0.7387152777777778]]]
exit_lane_num = [[4, 1], [2, 1], [4, 1], [2, 1]]
stop_lines = [[[0.4431610107421875, 0.2682291666666667], [0.4578094482421875, 0.2960069444444444]],
              [[0.4929656982421875, 0.2630208333333333], [0.5603485107421875, 0.2630208333333333]],
              [[0.6736297607421875, 0.3359375], [0.6589813232421875, 0.3602430555555556]],
              [[0.6462860107421875, 0.3949652777777778], [0.6501922607421875, 0.5234375]],
              [[0.6257781982421875, 0.8307291666666666], [0.6082000732421875, 0.8012152777777778]],
              [[0.5876922607421875, 0.8046875], [0.5134735107421875, 0.8012152777777778]],
              [[0.3992156982421875, 0.7248263888888888], [0.3845672607421875, 0.7526041666666666]],
              [[0.4138641357421875, 0.6901041666666666], [0.4109344482421875, 0.5442708333333334]]]  # 停止线位置
intersection_area = [[[0.4436492919921875, 0.2647569444444444], [0.4573211669921875, 0.2960069444444444],
                      [0.4895477294921875, 0.2682291666666667], [0.5959930419921875, 0.2612847222222222],
                      [0.6155242919921875, 0.2994791666666667], [0.6370086669921875, 0.2751736111111111],
                      [0.6741180419921875, 0.3307291666666667], [0.6565399169921875, 0.3567708333333333],
                      [0.6545867919921875, 0.6727430555555556], [0.6399383544921875, 0.7404513888888888],
                      [0.6536102294921875, 0.7630208333333334], [0.6262664794921875, 0.8272569444444444],
                      [0.6096649169921875, 0.8029513888888888], [0.4680633544921875, 0.8064236111111112],
                      [0.4446258544921875, 0.7942708333333334], [0.4280242919921875, 0.8098958333333334],
                      [0.3821258544921875, 0.7526041666666666], [0.3997039794921875, 0.7248263888888888],
                      [0.4133758544921875, 0.6970486111111112], [0.4065399169921875, 0.3932291666666667],
                      [0.4231414794921875, 0.3515625], [0.4075164794921875, 0.3116319444444444]]]

# # ---------------第二组测试数据---------------
# # 读取的txt数据
# txt_path = r'example\5.txt'
# # 底图图片
# image_path = r'example\5.jpg'
# # 超参
# threshold = 0.125
# min_cars = 5
#
# # 画面尺寸
# # w, h = (3810, 2160)
#
# mask = []  # 车道区域
# scale_line = [[0.4407196044921875, 0.2517361111111111], [0.5139617919921875, 0.2534722222222222]]  # 比例尺线
# scale_length = 5 * 3.5  # 比例尺的实际尺寸，以m为单位
# entrance_areas = [[[0.4368133544921875, 0], [0.4368133544921875, 0.2482638888888889],
#                    [0.5149383544921875, 0.25], [0.5129852294921875, 0]],
#                   [[0.6838836669921875, 0.4565972222222222], [0.6838836669921875, 0.5677083333333334],
#                    [0.9973602294921875, 0.5763888888888888], [0.9963836669921875, 0.4739583333333333]],
#                   [[0.5276336669921875, 0.8125], [0.6125946044921875, 0.828125],
#                    [0.6155242919921875, 0.9930555555555556], [0.5276336669921875, 0.9965277777777778]],
#                   [[0.0002899169921875, 0.5347222222222222], [0.1321258544921875, 0.5225694444444444],
#                    [0.3791961669921875, 0.5434027777777778], [0.3762664794921875, 0.6527777777777778],
#                    [0.0017547607421875, 0.6145833333333334]]]  # 进口道区域
# entrance_lane_num = [[1, 0, 3, 0, 1], [1, 0, 3, 0, 1], [3, 0, 3, 0, 1], [1, 0, 2, 0, 2]]
# exit_areas = [[[0.5198211669921875, 0.8098958333333334], [0.4514617919921875, 0.7977430555555556],
#                [0.4524383544921875, 0.9956597222222222], [0.5256805419921875, 0.9973958333333334]],
#               [[0.6848602294921875, 0.5824652777777778], [0.6790008544921875, 0.6796875],
#                [0.9954071044921875, 0.7057291666666666], [0.9963836669921875, 0.6085069444444444]],
#               [[0.5247039794921875, 0.2526041666666667], [0.5989227294921875, 0.24913194444444445],
#                [0.5959930419921875, 0.0008680555555555555], [0.5188446044921875, 0.0008680555555555555]],
#               [[0.0012664794921875, 0.4105902777777778], [0.3850555419921875, 0.4296875],
#                [0.3791961669921875, 0.5303819444444444], [-0.0006866455078125, 0.5026041666666666]]]
# exit_lane_num = [[5, 0], [4, 0], [5, 0], [4, 0]]
# stop_lines = [[[0.4407196044921875, 0.2517361111111111], [0.5139617919921875, 0.2534722222222222]],
#               [[0.6858367919921875, 0.421875], [0.6838836669921875, 0.5677083333333334]],
#               [[0.6223602294921875, 0.8263888888888888], [0.5256805419921875, 0.8107638888888888]],
#               [[0.3752899169921875, 0.6805555555555556], [0.3782196044921875, 0.5486111111111112]]]  # 停止线位置
# intersection_area = [[[0.4407196044921875, 0.2543402777777778], [0.3801727294921875, 0.3862847222222222],
#                       [0.3752899169921875, 0.6935763888888888], [0.4280242919921875, 0.7960069444444444],
#                       [0.6311492919921875, 0.8272569444444444], [0.6838836669921875, 0.7421875],
#                       [0.6848602294921875, 0.3845486111111111], [0.6096649169921875, 0.2560763888888889]]]

# ==================================================================
# ============================数据处理===============================
# ==================================================================

# # 读取result_txt的数据，进行后续处理
# with open(txt_path, 'r') as f:
#     lines = f.readlines()
# # 新建一个轨迹列表把这些数据储存起来
# tracks = []
# 遍历txt每一行数据
# for line in lines:
#     info_list = line.replace('\n', '').split('/')
#     # 获取属性里的每一个值
#     frame, id, x1, y1, x2, y2, conf, cls, track_cls, start_vector, end_vector = info_list
#     # 修改内部参数的属性并赋值道track中
#     track_info_dict = {'frame': int(frame), 'id': int(id), 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
#                        'conf': float(conf), 'cls': int(cls), 'track_cls': int(track_cls),
#                        'start_vector': eval(start_vector), 'end_vector': eval(end_vector)}
#     tracks.append(track_info_dict)
# print(tracks)

# 读取图片
img_pil = Image.open(image_path)
img_cv2 = np.array(img_pil)
img_base = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
# 宽高从这里拿
h, w = img_base.shape[:2]

# 执行绘图算法，并获取info_list
count_result, front_colors, info_list = draw_lines(img_base, txt_path, threshold=0.125, min_cars=5)
print('info_list第一条数据展示：', info_list[0])

# -----------------------------------------------
# Step1：去归一化，并转为numpy格式，方便计算，获取length_per_pixel
scale_line = np.array(scale_line) * np.array((w, h))
print('scale_line:', scale_line)
if len(scale_line) != 0 and scale_length:
    distance = calculate_distance(scale_line[0], scale_line[1])
    # 计算得到一个像素代表的实际真实长度（以m为单位）
    length_per_pixel = scale_length / distance
    print(length_per_pixel)

# get_mask(h, w, entrance_areas)

# intersection_mask = get_mask(h, w, intersection_area)
# speed = calculate_speed_at_intersection(info_list, length_per_pixel, intersection_mask)
# # 出口道
# exit_mask = get_each_mask(h, w, exit_areas)
# headway_times = calculate_headway_times(info_list, length_per_pixel, exit_mask, exit_lane_num, min_cars)
# headway_distances = calculate_headway_distances(info_list, length_per_pixel, exit_mask, exit_lane_num, min_cars)
stop_segments = unormalize_line(h, w, stop_lines)
calculate_queue_length(info_list, length_per_pixel, stop_segments, entrance_lane_num, h, w, entrance_areas)
