import sys

import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    最低代价线性分配
    Jonker-Volgenant算法比匈牙利算法运算速度快，因此此方法采用lap.lapjv()实现Jonker-Volgenant算法
    参数
    ----------
    cost_matrix : np.ndarray
        代价矩阵
    thresh : float
        最高阈值，默认设为0.8

    返回值
    ----------
    matches : [[track.idx, detection.idx], [track.idx, detection.idx],...]
        匹配上的track的idx和detection的idx
    unmatched_a : np.ndarray
        未匹配上的tracks的idx
    unmatched_b : np.ndarray
        未匹配上的detections的idx
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    # lap.lapjv()方法：
    # 参数
    # ------------
    # cost--传入一个N×N的代价矩阵；
    # extend_cost--如果传入的cost矩阵不是方阵则需要将这一项置为True；
    # cost_limit--线性匹配的最高阈值，超过这个值的不参加匹配
    #
    # 返回值
    # ------------
    # opt:分配过后的总cost，是一个float值
    # x: 每一个x与哪一个y匹配上了的idx，例:[2, 0, 1]指的是第一个x与第三个y匹配，第二个x与第一个y匹配，第三个x与第二个y匹配，为匹配成功的值为-1
    # y: 与x同理
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # 此处ix是匹配上的tracks的idx，mx是匹配上的detections的idx
    for ix, mx in enumerate(x):
        # >=0说明匹配上了
        if mx >= 0:
            # 将匹配上的track的idx和detection的idx加入matches
            matches.append([ix, mx])
    # 没有匹配上的tracks的idx
    unmatched_a = np.where(x < 0)[0]
    # 没有匹配上的detections的idx
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    根据IoU计算cost

    参数
    ----------
    atracks : list[tlbr]
        tracked状态和lost状态的tracks列表
    btracks : list[tlbr]
        detections的tracks列表

    返回值
    ----------
    ious : np.ndarray
        包含两个tracks的list中的元素两两之间的iou的矩阵
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    # bbox_ious是cython_bbox中的方法，输入是boxes_1: (N, 4) float矩阵, boxes_2: (K, 4) float矩阵
    # 输出是 (N, K) float矩阵
    # np.ascontiguousarray()将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    根据IoU计算cost，返回1-IoU作为代价矩阵

    参数
    ----------
    atracks : list[STrack]
        tracked状态和lost状态的tracks列表
    btracks : list[STrack]
        detections的tracks列表

    返回值
    ----------
    cost_matrix : np.ndarray
        代价矩阵
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 此处调用STrack类中的tlbr方法，tlbr方法中再调用tlwh方法，将strack._tlwh转换为tlbr
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """
    根据IoU计算cost没有考虑到检测框置信度的影响，此方法是为了加入检测框分数的影响
    通过将1-cost_matrix乘以检测框分数得到fuse_sim，则1-fuse_sim就是加入了检测框分数权重的cost_matrix

    参数
    ----------
    cost_matrix : np.ndarray
        tracked状态和lost状态的tracks列表
    detections : list[STrack]
        detections的tracks列表

    返回值
    ----------
    fuse_cost : np.ndarray
        融合后的代价矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    # 将detections的分数提出来放进数组，如果有三个检测框，则shape为[score, score, score]
    det_scores = np.array([det.score for det in detections])
    # np.expand_dims: [[score, score, score]]  ------------>
    # repeat(cost_matrix.shape[0]就等于tracked状态和lost状态的tracks的数目，此处假设是4个): [[score, score, score],
    #                                                                                 [score, score, score],
    #                                                                                 [score, score, score],
    #                                                                                 [score, score, score]]
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # 按照假设，有3个detections和4个现成的tracks，则iou_sim的shape为[4,3]，det_scores的shape也为[4,3],可以对位相乘得到fuse_sim
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
