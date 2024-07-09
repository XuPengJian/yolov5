import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import sys
from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState


# 定义track类
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls):

        # 等待激活的track
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod  # 静态方法: 可以不用实例化直接STrack.multi_predict()调用此方法
    def multi_predict(stracks):
        """
        通过卡尔曼滤波预测当前位置

        参数
        ----------
        stracks : List[STrack1, STrack2.....]
            tracks列表，包含tracked状态和lost状态的tracks

        """
        if len(stracks) > 0:
            # multi_mean的结构为[[x, y, a, h, vx, vy, va, vh],
            #                  [x, y, a, h, vx, vy, va, vh],...]
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            # multi_convariance的shape为[n,8,8]
            multi_covariance = np.asarray([st.covariance for st in stracks])
            # 遍历所有传入的tracks
            for i, st in enumerate(stracks):
                # 如果tracks的状态不是Tracked，即状态为lost，则将vh设为0  why？？？
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            # 通过卡尔曼滤波预测当前位置
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            # 将tracks的mean和covariance属性更新为预测结果
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov



    def activate(self, kalman_filter, frame_id):
        """
        当第一次检测到目标时，激活track，但是此时track.is_activated不设为True(除非是第二帧？)
        此方法说是激活，其实就是初始化，并不算真正意义上的激活

        参数
        ----------
        kalman_filter : kalman_filter.KalmanFilter
            卡尔曼滤波
        frame_id : int
            当前帧的id

        """
        self.kalman_filter = kalman_filter
        # id + 1
        self.track_id = self.next_id()
        # 初始化用于卡尔曼滤波的mean和covariance
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        #
        self.tracklet_len = 0
        # 将track的状态设为Tracked，track的初始状态是new，在basetrack中定义
        self.state = TrackState.Tracked
        # track中的frame_id和tracker中的frame_id不一样，track中的是其最后一次匹配上的那一帧的id，tracker中的是当前的id
        # 此处传入的frame_id是tracker的，所以是当前帧的id
        # 如果是视频的第二帧，则将track设置为已激活状态    why？？？
        if frame_id == 1:
            self.is_activated = True
        # 将当前帧的id赋给track.frame_id
        self.frame_id = frame_id
        # track.start_frame表示从那一帧开始被激活的
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        用于重新激活状态为lost的tracks

        参数
        ----------
        new_track : STrack
            第二次匹配成功的track
        frame_id : int
            当前帧的id
        new_id : bool
            当前帧的id
        """
        # 更新卡尔曼滤波的均值和协方差
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        # 初始化视频序列长度为0，此处的视频序列指的是这个track的序列
        self.tracklet_len = 0
        # 将track的状态设为Tracked
        self.state = TrackState.Tracked
        # 将track.is_activated设为True
        self.is_activated = True
        # 将当前的帧id赋给track.frame_id
        self.frame_id = frame_id
        # 如果new_id为True，则track的id+1
        if new_id:
            self.track_id = self.next_id()
        # 将
        self.score = new_track.score
    # update和re-activate都包含了卡尔曼滤波的更新,这是因为这两个方法不会同时使用，tracked状态的tracks用update方法
    # lost状态的tracks用re-activate方法
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# 定义tracker类
class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        # tracked状态的tracks type: list[STrack]
        self.tracked_stracks = []
        # lost状态的tracks type: list[STrack]
        self.lost_stracks = []
        # removed状态的tracks type: list[STrack]
        self.removed_stracks = []
        # 当前帧的id
        self.frame_id = 0
        # 超参
        self.args = args
        # 两轮匹配结束后，剩下未匹配成功的detections的分数需要大于det_thresh这个阈值才能被初始化为新的track
        # 超参中设置为0.5，即此处设置det_thresh为0.6
        self.det_thresh = args.track_thresh + 0.1
        # frame_rate是视频的帧率，此处取30，args.track_buffer是超参中设置的最大lost值，超出这个值的track就会被remove
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        # 将buffer_size的值赋给max_time_lost
        self.max_time_lost = self.buffer_size
        # 实例化卡尔曼滤波
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        """
        tracker的更新方法

        参数
        ----------
        output_results : tensor([[xyxy, conf, cls],
                                 [xyxy, conf, cls],...]
            yolov5+NMS的结果
        img_info : [h(int), w(int)]
            视频帧的高和宽
        """
        # 帧id加1
        self.frame_id += 1
        # 用于存储已激活的tracks的List
        activated_starcks = []
        # 用于存储重新找到的tracks的List
        refind_stracks = []
        # 用于存储丢失的tracks的List
        lost_stracks = []
        # 用于存储被移除的tracks的List
        removed_stracks = []
        # 如果检测结果是[xyxy, conf]，没有cls(这个情况不考虑了!)
        # if output_results.shape[1] == 5:
        #     scores = output_results[:, 4]
        #     bboxes = output_results[:, :4]
        #     classes = output_results[:, 5]
        # 如果检测结果是[xyxy, conf, cls]

        # output_results = output_results.cpu().numpy()
        # 本身就是numpy格式不用再转了
        output_results = output_results
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]  # x1y1x2y2
        classes = output_results[:, 5]
        # 获取视频帧的高和宽，但是并没有用到
        img_h, img_w = img_info[0], img_info[1]

        # 筛选分数大于track_thresh的检测框，即高分框
        # 假如有四个框，则scores的结构是[scores1, scores2, scores3, scores4],分别是四个框的分数
        # 如果前三个框的分数大于阈值，最后一个小于阈值，则 remain_inds = [True, True, True, False]
        remain_inds = scores > self.args.track_thresh
        # 筛选出分数在0.1到track_thresh的检测框，即低分框
        # 0.1为低分框的阈值下限
        inds_low = scores > 0.1
        # track_thresh为低分框的阈值上限
        inds_high = scores < self.args.track_thresh
        # 通过逻辑与运算得出低分框的idx
        inds_second = np.logical_and(inds_low, inds_high)
        # 通过idx获取低分框的xyxy
        dets_second = bboxes[inds_second]
        # 通过idx获取高分框的xyxy
        dets = bboxes[remain_inds]
        # 通过idx获取高分框的分数
        scores_keep = scores[remain_inds]
        # 通过idx获取低分框的分数
        scores_second = scores[inds_second]
        # 通过idx获取低分框的类别
        cls_second = classes[inds_second]
        # 通过idx获取高分框的类别
        cls = classes[remain_inds]
        # 如果高分框的List不为空，则将检测结果bbox的tlbr格式转换为tlwh格式，并和分数一起传给STrack类初始化一个track实例
        # 再将tracks加入detections列表
        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, cls)]
        # 如果高分框的List为空，则定义detections为空列表
        else:
            detections = []

        ''' 将刚刚被检测到的tracks加入tracked_stracks'''
        # 用于存储是tracked状态但是没有被激活的tracks，也就是只被检测到一次
        unconfirmed = []
        # 用于存储是tracked状态并且已经被激活的tracks
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' 第一次匹配, 将高分框与所有tracks(状态为Tracked且is_activated==True和Lost)进行匹配'''
        # 将tracked状态且已经被激活的的tracks和lost状态的tracks合并
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # 根据卡尔曼滤波预测当前位置
        STrack.multi_predict(strack_pool)
        # 通过iou_distance方法计算tracks(tracked和lost状态)和detections两两track之间的iou
        # 返回的cost matrix是1-iou的结果
        dists = matching.iou_distance(strack_pool, detections)
        # 如果不是Mot20数据集，则采用融合了detection分数的cost_matrix
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # 线性分配计算，得到：匹配上的track的idx和detection的idx，未匹配上的tracks的idx，未匹配上的detections的idx
        # 设置的默认匹配阈值是0.8
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # 从matches中获取track的idx和detection的idx
        for itracked, idet in matches:
            # 获取匹配上的tracks
            track = strack_pool[itracked]
            # 获取匹配上的detections
            det = detections[idet]
            # 如果匹配上的tracks的状态是Tracked，则将track进行update并将其加入activated_starcks
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)

            # 如果匹配上的tracks的状态是lost，则将track进行re_activate并将其加入refind_stracks
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' 第二次匹配，将低分框与剩下的tracks进行匹配'''
        # 如果低分框列表不为空
        if len(dets_second) > 0:
            # 将bbox的tlbr格式转换为tlwh格式，并和分数一起传给STrack类初始化一个track实例，再将tracks加入detections列表
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets_second, scores_second, cls_second)]
        # 如果低分框列表为空，则将detections_second置为空列表
        else:
            detections_second = []
        # 获取第一次匹配中没有匹配上的tracks，注意！！此时仅获取状态为Tracked的track，此处与第一次匹配中既获取Tracked也获取lost不同
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 将剩下的tracks与低分框进行一一计算iou，得到代价矩阵
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 得到：匹配上的track的idx和detection的idx，未匹配上的tracks的idx，未匹配上的detections的idx
        # 注意！！此处阈值设的较低，为0.5
        # 注意！！！此处的低分框仅用于此次匹配，如果没有匹配上则在之后并不会被用到
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        #  从matches中获取track的idx和detection的idx
        for itracked, idet in matches:
            # 获取匹配上的tracks
            track = r_tracked_stracks[itracked]
            # 获取匹配上的detections
            det = detections_second[idet]
            # 如果匹配上的tracks的状态是Tracked，则将track进行update并将其加入activated_starcks
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            # 如果匹配上的tracks的状态是lost，则将track进行re_activate并将其加入refind_stracks
            # 但是匹配时仅筛选出了Tracked状态的tracks，所以此处没有lost状态的tracks，因此不会执行else
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # 第二次匹配中还是没匹配上的tracks
        for it in u_track:
            # 通过idx获取tracks
            track = r_tracked_stracks[it]
            # 如果track的状态不是lost则将其置为lost并加入lost_stracks
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''最后再将unconfirmed状态的tracks(即那些只被检测到一次的track)与第一次匹配剩下的高分框进行匹配，将没匹配上的tracks移除'''
        # 获取第一次匹配没匹配上的detections
        detections = [detections[i] for i in u_detection]
        # 将unconfirmed的tracks与剩下的高分框进行一一计算iou，得到代价矩阵
        dists = matching.iou_distance(unconfirmed, detections)
        # 如果不是Mot20数据集，则采用融合了detection分数的cost_matrix
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # 得到：匹配上的track的idx和detection的idx，未匹配上的tracks的idx，未匹配上的detections的idx
        # 设置的阈值是0.7，与第一次匹配相近
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # 从matches中获取track的idx和detection的idx
        for itracked, idet in matches:
            # 将匹配上的原先为unconfirmed的tracks进行update，update方法会将其is_activated属性设为True
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            # 将track添加进activated_starcks
            activated_starcks.append(unconfirmed[itracked])
        # 此次没匹配上的原先为unconfirmed的tracks，将其标记为removed并加入removed_stracks
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ 将未匹配上的detections初始化新的tracks"""
        # 获取未匹配上的高分框的tracks的idx
        for inew in u_detection:
            # 获取未匹配上的高分框的tracks
            track = detections[inew]
            # 两轮匹配结束后，剩下未匹配成功的detections的分数需要大于det_thresh这个阈值才能被初始化为新的track
            # 默认设置为0.6
            if track.score < self.det_thresh:
                continue
            # 初始化track
            # 注意！！此时虽然是执行了activate方法，但是track的is_activated属性依然是False，也就是track依然是处于unconfirmed状态
            track.activate(self.kalman_filter, self.frame_id)
            # 将初始化后的tracks加入activated_starcks
            activated_starcks.append(track)
        """ 将状态为lost并超过一定帧数的tracks标记为removed"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 将状态为Tracked的tracks赋值给tracker.tracked_stracks
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 合并tracker.tracked_stracks和activated_starcks
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 合并tracker.tracked_stracks和refind_stracks
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 如果一个track既在lost_stracks，也在tracked_stracks中，则把这个track从lost_stracks中删除
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 此处使用了列表的extend()方法，将lost_stracks加入lost_stracks
        self.lost_stracks.extend(lost_stracks)
        # 如果一个track既在lost_stracks，也在removed_stracks中，则把这个track从lost_stracks中删除
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 将removed_stracks加入tracker.removed_stracks
        self.removed_stracks.extend(removed_stracks)
        # 将tracked_stracks和lost_strack中tracks两两计算iou，取大于0.85的tracks对比谁存在的更久，保留存在更久的那一个，另一个删除
        # 此步骤可能是为了防止本来应该是同一个track但是发生了id switch，所以就只保留存在更久的那一个
        # 但是这样可能就会发生id的跳跃？比如id=12的track(lost状态)与id=20的track(Tracked状态)的iou大于0.85，
        # 则会直接跳过20这个id，导致下一个id从21开始
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # 输出在tracked_stracks中并且is_activated是True的那些tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        # 输出的是各种状态的tracks
        return output_stracks

# 合并两个存储tracks的List，建了exists字典来防止重复添加
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

# alist中的元素如果blist中也有，则删除，此方法判断条件是id
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        # 此处使用的字典的get()方法，如果字典中已经存在了tid这个key则返回其对应的value，否则返回0
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    # 将stracksa中的tracks与将stracksb中的tracks一一计算iou，得到代价矩阵
    pdist = matching.iou_distance(stracksa, stracksb)
    # 得到代价小于0.15的匹配，返回一个元组([],[])，第一个数组是行的idx，第二个数组是列的idx
    # 例: 得到([0, 0, 0, 1], [0, 1, 2, 0])，则表示(0,0),(0,1),(0,2),(1,0)这四个cost matrix中的元素符合小于0.15的条件
    pairs = np.where(pdist < 0.15)
    # 初始化两个list
    dupa, dupb = list(), list()
    # p, q分别是符合条件的stracksa，stracksb的索引
    for p, q in zip(*pairs):
        # 根据索引得到stracksa中的stracks的存在帧数(当前帧id减第一次被检测到的帧id)
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        # 根据索引得到stracksb中的stracks的存在帧数(当前帧id减第一次被检测到的帧id)
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        # 如果a中的track存在时间大于b中的track，则将b中的索引加入dupb，否则将a中的索引加入dupa
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    # 将索引不在dupa和dupb中的tracks分别加入resa和resb
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
