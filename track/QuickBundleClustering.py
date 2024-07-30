import numpy as np
import math
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from dipy.segment.clustering import QuickBundles

# from dipy.viz import colormap

"""
平面向量归一化长度为1
"""


def normalize(vector):
    """
    :param vector: (x,y) numpy格式
    :return:
    """
    x = vector[0]
    y = vector[1]
    # todo:分母为0的情况欠考虑
    vector /= math.sqrt(x ** 2 + y ** 2)
    return vector


def cal_distance(pt1, pt2):
    d_x = pt1[0] - pt2[0]
    d_y = pt1[1] - pt2[1]
    distance = math.sqrt(d_x ** 2 + d_y ** 2)
    return distance


def cal_trajectory_length(traj):
    length = 0.0
    for i in range(len(traj) - 1):
        length += cal_distance(traj[i], traj[i + 1])
    return length


def cal_unit_direction_vector(traj):
    """
    计算轨迹中每一段的单位方向向量
    :param traj: 轨迹列表，包含一系列的点，每个点由(x, y)坐标组成
    :return: 单位方向向量列表，每个向量由(x, y)组成
    """
    unit_vectors = []
    for i in range(len(traj) - 1):
        start_point = traj[i]
        end_point = traj[i + 1]

        # 计算方向向量
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # 计算向量的长度
        vector_length = (dx ** 2 + dy ** 2) ** 0.5
        print(vector_length)

        # 避免除以零的情况
        # 这里可以修改成某个阈值（小于多少个比例范围认为是不动的千分之一？或1/500），用来判断车是否处于静止状态
        if vector_length == 0:
            unit_vectors.append('静止')
            continue

        # 计算单位向量
        unit_vector = (dx / vector_length, dy / vector_length)

        # 将单位向量添加到列表中
        unit_vectors.append(unit_vector)

    # 最后一位向量用最后的那个值
    unit_vectors.append(unit_vectors[-1])
    return unit_vectors


def resample_points(trajectory, num_points):
    '''
    Args:
        trajectory: 轨迹，是一个list: [[x,y], [x,y]...]
    Returns:
        重新采样的点，也是一个list: [[x,y], [x,y]...]
    '''
    points = np.array(trajectory)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    distance, index = np.unique(distance, return_index=True)
    splines = [UnivariateSpline(distance, coords, k=3, s=0) for coords in points[index].T]
    alpha = np.linspace(np.min(distance), np.max(distance), num_points)
    points_fitted = np.vstack([spl(alpha) for spl in splines]).T
    return points_fitted


"""
新轨迹获取,考虑延长线的方向,只取其中一端
"""


def extend_lines(traj):
    # 得到延长起始点 pt1->pt2(延长的方向)
    pt1 = traj[1]
    pt2 = traj[0]
    pt3, pt4 = extend_line(pt1, pt2)
    base_vector = pt2 - pt1
    vector = pt3 - pt1
    if sum(base_vector * vector) > 0:
        new_start_pt = pt3
    else:
        new_start_pt = pt4
    # 得到延长结束点 pt1->pt2(延长的方向)
    pt1 = traj[-2]
    pt2 = traj[-1]
    pt3, pt4 = extend_line(pt1, pt2)
    base_vector = pt2 - pt1
    vector = pt3 - pt1
    if sum(base_vector * vector) > 0:
        new_end_pt = pt3
    else:
        new_end_pt = pt4

    # 获取等分点(为了让曲线更受控制，拥有更合理的控制点)
    new_start_pt_list = []
    new_end_pt_list = []
    # 基于每段的resample长度作为延长线段的等分数量(这样更合理)
    each_length = math.sqrt(sum((traj[0] - traj[1]) ** 2))
    new_start_extend_length = math.sqrt(sum((traj[0] - new_start_pt) ** 2))
    new_end_extend_length = math.sqrt(sum((new_end_pt - traj[-1]) ** 2))
    # 起点等分份数基于长度计算
    n = int(new_start_extend_length / each_length + 1)
    # 起点等分(包括新开始点,不包括结束点,结束点一开始就存在)
    for i in range(n):
        each_point = new_start_pt + i * (traj[0] - new_start_pt) / n
        new_start_pt_list.append(each_point)
    # 终点等分份数基于长度计算
    n = int(new_end_extend_length / each_length + 1)
    # 终点等分(不包括开始点,包括结束点,开始点一开始就存在)
    for i in range(1, n + 1):
        each_point = traj[-1] + i * (new_end_pt - traj[-1]) / n
        new_end_pt_list.append(each_point)

    # 将新的起点和新的终点加入到初始轨迹
    traj = np.array(new_start_pt_list + list(traj) + new_end_pt_list)
    # traj = np.insert(traj, 0, new_start_pt, axis=0)
    # traj = np.insert(traj, -1, new_end_pt, axis=0)
    # traj[0] = new_start_pt
    # traj[-1] = new_end_pt
    return traj


"""
延长线,不考虑向量方向,两端都延长
"""


# 这里默认考虑边界的最大值为(1,1)
def extend_line(p1, p2, SCALE=100):
    # Calculate the intersection point given (x1, y1) and (x2, y2)
    # 输出的坐标为numpy格式
    def line_intersection(line1, line2):
        x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def detect(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = detect(x_diff, y_diff)
        if div == 0:
            raise Exception('lines do not intersect')

        dist = (detect(*line1), detect(*line2))
        x = detect(dist, x_diff) / div
        y = detect(dist, y_diff) / div
        return x, y

    # 做初始化值
    x1, x2 = 0, 0
    y1, y2 = 0, 0

    # Reorder smaller X or Y to be the first point  小值点作为p1
    # and larger X or Y to be the second point  大值点作为p2
    try:
        slope = (p2[1] - p1[1]) / (p1[0] - p2[0])  # 计算斜率  (y2-y1)/(x1-x2)
        # HORIZONTAL or DIAGONAL  # 水平or倾斜
        if p1[0] <= p2[0]:
            x1, y1 = p1
            x2, y2 = p2
        else:
            x1, y1 = p2
            x2, y2 = p1
    # 如果斜率不存在
    except ZeroDivisionError:
        # VERTICAL
        # 直接比较y值大小
        # 小的做p1, 大的做p2
        if p1[1] <= p2[1]:
            x1, y1 = p1
            x2, y2 = p2
        else:
            x1, y1 = p2
            x2, y2 = p1

    # Extend after end-point A
    length_A = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    p3_x = x1 + (x1 - x2) / length_A * SCALE
    p3_y = y1 + (y1 - y2) / length_A * SCALE

    # Extend after end-point B
    length_B = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    p4_x = x2 + (x2 - x1) / length_B * SCALE
    p4_y = y2 + (y2 - y1) / length_B * SCALE

    # --------------------------------------
    # Limit coordinates to borders of image
    # --------------------------------------
    # HORIZONTAL
    if y1 == y2:
        if p3_x < 0:
            p3_x = 0
        if p4_x > 1:
            p4_x = 1
        return (np.array((p3_x, p3_y)), np.array((p4_x, p4_y)))
    # VERTICAL
    elif x1 == x2:
        if p3_y < 0:
            p3_y = 0
        if p4_y > 1:
            p4_y = 1
        return (np.array((p3_x, p3_y)), np.array((p4_x, p4_y)))
    # DIAGONAL
    else:
        A = (p3_x, p3_y)
        B = (p4_x, p4_y)

        C = (0, 0)  # C-------D
        D = (1, 0)  # |-------|
        E = (1, 1)  # |-------|
        F = (0, 1)  # F-------E

        if slope > 0:
            # 1st point, try C-F side first, if OTB then F-E
            new_x1, new_y1 = line_intersection((A, B), (C, F))
            if new_x1 > 1 or new_y1 > 1:
                new_x1, new_y1 = line_intersection((A, B), (F, E))

            # 2nd point, try C-D side first, if OTB then D-E
            new_x2, new_y2 = line_intersection((A, B), (C, D))
            if new_x2 > 1 or new_y2 > 1:
                new_x2, new_y2 = line_intersection((A, B), (D, E))

            return (np.array((new_x1, new_y1)), np.array((new_x2, new_y2)))
        elif slope < 0:
            # 1st point, try C-F side first, if OTB then C-D
            new_x1, new_y1 = line_intersection((A, B), (C, F))
            if new_x1 < 0 or new_y1 < 0:
                new_x1, new_y1 = line_intersection((A, B), (C, D))
            # 2nd point, try F-E side first, if OTB then E-D
            new_x2, new_y2 = line_intersection((A, B), (F, E))
            if new_x2 > 1 or new_y2 > 1:
                new_x2, new_y2 = line_intersection((A, B), (E, D))
            return (np.array((new_x1, new_y1)), np.array((new_x2, new_y2)))


# theta = np.linspace(-3, 2, 40)
# points = np.vstack((np.cos(theta), np.sin(theta))).T
# print(points)
if __name__ == '__main__':
    from dipy.data import get_fnames
    from dipy.io.streamline import load_tractogram
    from dipy.tracking.streamline import Streamlines

    # fname = get_fnames('fornix')
    # fornix = load_tractogram(fname, 'same', bbox_valid_check=False).streamlines
    # streamlines = Streamlines(fornix)
    # print(streamlines)
