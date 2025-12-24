# Ultralytics YOLO 🚀, AGPL-3.0 license

import importlib.util
import math
from collections import Counter, deque

import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC        ## 导入全局运动补偿（GMC）类，处理相机运动导致的目标偏移
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """
    An extended version of the STrack class for YOLOv8, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features(feat): Update features vector and smooth it using exponential moving average.
        predict(): Predicts the mean and covariance using Kalman filter.
        re_activate(new_track, frame_id, new_id): Reactivates a track with updated features and optionally new ID.
        update(new_track, frame_id): Update the YOLOv8 instance with new track and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(tlwh=[100, 50, 80, 40], score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(tlwh=[110, 60, 80, 40], score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50, reid_conf=None):
        """
        Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            tlwh (np.ndarray): Bounding box coordinates in tlwh format (top left x, top left y, width, height).
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.          目标类别的ID，还有种类在哪里
            feat (np.ndarray | None): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.

        Examples:
            Initialize a BOTrack object with bounding box, score, class ID, and feature vector
            >>> tlwh = np.array([100, 50, 80, 120])
            >>> score = 0.9
            >>> cls = 1
            >>> feat = np.random.rand(128)
            >>> bo_track = BOTrack(tlwh, score, cls, feat)
        """
        super().__init__(tlwh, score, cls)         #继承父类STrack的方法  

        self.smooth_feat = None     # 平滑后的特征向量（用于更稳定的外观匹配）
        self.curr_feat = None       # 当前帧的特征向量（最新提取的外观特征）
        if feat is not None:        #
            self.update_features(feat)   # 若有初始特征，立即更新
        self.features = deque([], maxlen=feat_history)   # 特征历史队列（限制最大长度，默认50）
        self.alpha = 0.9   # 特征平滑的指数移动平均因子（0.9表示更依赖历史特征）

        self.reid_conf = reid_conf
        self.cls_history = deque()
        self.perm_id = None
        self.id_locked = False
        self.label_votes = deque(maxlen=15)
        self.fish_label = -1
        self.is_temporary = True
        self.hits = 0
        # 速度记录，仅用于调试（本版本未在关联中使用）
        self.prev_center = None
        self.speed = 0.0

    def update_features(self, feat):
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """Predicts the object's future state using the Kalman filter to update its mean and covariance."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
    ##################################################
    def activate(self, kalman_filter, frame_id):
        """Activates a new tracklet and records its initial classification for later voting."""
        if self.cls is not None:
            self.cls_history.append(int(self.cls))
        super().activate(kalman_filter, frame_id)
        self.hits = 1
        self._update_speed()
    ###################################################################
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        #add
        if new_track.cls is not None:
            self.cls_history.append(int(new_track.cls))

        super().re_activate(new_track, frame_id, new_id)
        #add
        self.hits += 1
        self._update_speed()

    def update(self, new_track, frame_id):
        """Updates the YOLOv8 instance with new track information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        ##add
        if new_track.cls is not None:
            self.cls_history.append(int(new_track.cls))
        super().update(new_track, frame_id)
        self.hits += 1
        self._update_speed()

    def add_label_vote(self, label: int):
        """Accumulate ReID classifier votes and keep the majority as fish_label."""
        if label is None:
            return
        label = int(label)
        if label < 0:
            return
        self.label_votes.append(label)
        ctr = Counter(self.label_votes)
        if len(ctr) > 0:
            lbl, _ = ctr.most_common(1)[0]
            self.fish_label = int(lbl)

    def promote_to_permanent(self, perm_id: int):
        """Mark this track as a permanent track with a fixed perm_id."""
        self.is_temporary = False
        self.perm_id = int(perm_id)
        self.id_locked = True

    def _update_speed(self):
        """Update per-frame speed estimate based on bbox center displacement."""
        tlwh = self.tlwh
        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]
            self.speed = math.hypot(dx, dy)
        self.prev_center = (cx, cy)

    @property  #@property 使其可像属性一样访问（track.tlwh）
    def tlwh(self):
        """Returns the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Converts tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def result(self):
        """返回跟踪结果，优先使用每帧分配好的 display_id，并附带原始 track_id。"""
        coords = self.xyxy if self.angle is None else self.xywha

        # 优先使用 BOTSORT 在当前帧分配好的 display_id，保证一帧内鱼号唯一
        display_id = getattr(self, "display_id", None)
        if display_id is None:
            # 回退策略：perm_id > fish_label+1 > track_id
            display_id = int(self.track_id)
            if self.perm_id is not None:
                display_id = int(self.perm_id)
            elif self.fish_label is not None and self.fish_label >= 0:
                display_id = int(self.fish_label) + 1
        else:
            display_id = int(display_id)

        perm = -1 if self.perm_id is None else int(self.perm_id)
        return coords.tolist() + [display_id, self.score, self.cls, self.idx, perm, int(self.track_id)]



class BOTSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter(): Returns an instance of KalmanFilterXYWH for object tracking.
        init_track(dets, scores, cls, img): Initialize track with detections, scores, and classes.
        get_dists(tracks, detections): Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict(tracks): Predict and track multiple objects with YOLOv8 model.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(dets, scores, cls, img)
        >>> bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with the YOLOv8 object detection model and supports ReID only if enabled via args.
    """

    N_INIT_FRAMES = 15
    N_VOTE_INIT = 5

    def __init__(self, args, frame_rate=15):
        """
        Initialize YOLOv8 object with ReID module and GMC algorithm.

        Args:
            args (object): Parsed command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video being processed.

        Examples:
            Initialize BOTSORT with command-line arguments and a specified frame rate:
            >>> args = parse_args()
            >>> bot_sort = BOTSORT(args, frame_rate=30)
        """
        super().__init__(args, frame_rate)   
        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        
        self.encoder = None
        self.used_perm_ids = set()
        self.permanent_tracks = {}
        self.MAX_FISH = 9
        self.stable_frames = getattr(args, "stable_frames", 8)   ##可调参
        self.min_votes = getattr(args, "min_votes", 4)           ##可调参
        self.takeover_hits_margin = getattr(args, "takeover_hits_margin", 5)

        if getattr(args, "with_reid", False):
            weights = getattr(args, "reid_weights", None)
            if weights:
                spec = importlib.util.find_spec("reid.extract_embeddings")
                if spec is None:
                    LOGGER.warning("ReID module not found; continuing without ReID support.")
                else:
                    from reid.extract_embeddings import ReIDEncoder

                    device = getattr(args, "reid_device", "cuda")
                    try:
                        self.encoder = ReIDEncoder(weights, device=device)
                        LOGGER.info(f"Loaded ReIDEncoder from {weights} on {device}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to load ReID weights {weights}: {e}")
            else:
                LOGGER.warning("with_reid is enabled but no reid_weights provided; disabling ReID.")

        # if args.with_reid:
        #     # Haven't supported BoT-SORT(reid) yet
        #     self.encoder = None
        self.gmc = GMC(method=args.gmc_method)
    
    def update(self, results, img=None):
        """
        Run one tracking step, then assign per-frame unique display IDs:
        1) perm_id 第一优先；
        2) 每个 fish_label 至多一条轨迹用 fish_label+1 作为显示 ID；
        3) 其他轨迹回退使用自身唯一的 track_id。
        """
        if img is not None:
            self.img_h, self.img_w = img.shape[:2]

        # 让 BYTETracker 完成标准的关联、卡尔曼预测等内部更新
        super().update(results, img)

        # 先更新永久代表轨迹（S4），再做本帧的 display_id 分配（S5）
        self._update_permanent_tracks()
        self._assign_display_ids()

        # 使用更新后的 display_id 重新组装输出
        outputs = []
        for track in self.tracked_stracks:
            if track.is_activated:
                outputs.append(track.result)

        if len(outputs) == 0:
            # 返回 10 列：x1,y1,x2,y2,display_id,score,cls,idx,perm_id,track_id
            return np.zeros((0, 10), dtype=float)
        return np.asarray(outputs, dtype=float)

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(dets) == 0:
            return []
        # if self.args.with_reid and self.encoder is not None:
        #     features_keep = self.encoder.inference(img, dets)
        #     return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]  # detections
        # else:
        #     return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]  # detections
        features_keep, reid_labels, reid_confs = None, None, None
        if self.encoder is not None and img is not None:
            # dets are xywh(+idx); convert to xyxy for cropping
            xyxy = xywh2xyxy(np.asarray(dets)[:, :4])
            try:
                features_keep, reid_labels, reid_confs = self.encoder.encode_and_classify(img, xyxy)
            except AttributeError:
                # Fallback for encoders without encode_and_classify
                features_keep = self.encoder(img, xyxy)
                try:
                    reid_labels, reid_confs = self.encoder.predict_labels(img, xyxy)
                except Exception:
                    reid_labels, reid_confs = None, None
            except Exception as e:
                LOGGER.warning(f"ReID feature extraction failed: {e}")

        tracks = []
        for i, (xywh, s, c) in enumerate(zip(dets, scores, cls)):
            feat = features_keep[i] if features_keep is not None and len(features_keep) > i else None
            label = reid_labels[i] if reid_labels is not None and len(reid_labels) > i else c
            conf = reid_confs[i] if reid_confs is not None and len(reid_confs) > i else None
            track = BOTrack(xywh, s, label, feat=feat, reid_conf=conf)
            if reid_labels is not None and len(reid_labels) > i:
                track.add_label_vote(reid_labels[i])
            tracks.append(track)
        return tracks

    def get_dists(self, tracks, detections):
        #"""Calculates distances between tracks and detections using IoU and optionally ReID embeddings."""
        """
        仅使用原生的 IoU + ReID 距离（无 S2、无 S3），体现 S1+S4 的消融设定。
        """
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        #if self.args.fuse_score:
        if getattr(self.args, "fuse_score", False):
            dists = matching.fuse_score(dists, detections)

        #if self.args.with_reid and self.encoder is not None:
        use_reid = getattr(self.args, "with_reid", False) and self.encoder is not None
        if use_reid and len(tracks) > 0 and len(detections) > 0:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks):
        """Predicts the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """Resets the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self.gmc.reset_params()
        
        self.used_perm_ids = set()
        self.permanent_tracks = {}

    def _maybe_lock_ids(self):
        """Assigns permanent IDs via majority voting during the initial warmup frames."""
        if self.frame_id > self.N_INIT_FRAMES:
            return

        for track in self.tracked_stracks:
            if getattr(track, "id_locked", False):
                continue

            if len(getattr(track, "cls_history", [])) >= self.N_VOTE_INIT:
                voted_id = Counter(track.cls_history).most_common(1)[0][0]
                if 1 <= voted_id <= 9 and voted_id not in self.used_perm_ids:
                    track.perm_id = int(voted_id)
                    track.id_locked = True
                    self.used_perm_ids.add(track.perm_id)

    def _assign_display_ids(self):
        """
        S5：每帧为所有轨迹分配唯一的显示 ID。
        规则：
          1) perm_id（1..9）优先且唯一；
          2) 对于 perm_id 为空但 fish_label 稳定的临时轨迹，
             每个 fish_label 至多选一个代表，显示为 fish_label+1；
          3) 其他轨迹统一回退显示为自身的 track_id（天然唯一）。
        这样就不会出现“一帧里两个 6 号鱼”的情况。
        """
        if not self.tracked_stracks:
            return

        # 先清空之前的 display_id
        for t in self.tracked_stracks:
            if hasattr(t, "display_id"):
                t.display_id = None

        used_ids = set()

        # 标记本帧是否已经有 perm_id 或稳定 fish_label 候选
        has_perm = False

        # 1) perm_id 第一优先，占用 1..9 中的号码
        for t in self.tracked_stracks:
            perm = getattr(t, "perm_id", None)
            if perm is None or perm <= 0:
                continue
            has_perm = True
            did = int(perm)
            if did in used_ids:
                continue
            t.display_id = did
            used_ids.add(did)

        # 2) 为没有 perm_id、但 fish_label 稳定的临时轨迹，挑选代表
        candidates_by_fish = {k: [] for k in range(self.MAX_FISH)}  # 0..8
        has_stable_candidate = False
        for t in self.tracked_stracks:
            if getattr(t, "display_id", None) is not None:
                continue

            fish_label = getattr(t, "fish_label", None)
            if fish_label is None or fish_label < 0:
                continue

            k = int(fish_label)
            if not (0 <= k < self.MAX_FISH):
                continue

            label_votes = getattr(t, "label_votes", None)
            try:
                support = Counter(label_votes).get(fish_label, 0) if label_votes is not None else 0
            except Exception:
                support = 0

            hits = getattr(t, "hits", 0)
            conf = getattr(t, "reid_conf", 0.0) or 0.0
            det_score = getattr(t, "score", 0.0) or 0.0

            if hits < self.stable_frames or support < self.min_votes:
                continue

            has_stable_candidate = True
            score = (hits, support, conf, det_score)
            candidates_by_fish[k].append((score, t))

        # ---- 纯 warmup 阶段：既没有 perm_id，也没有任何稳定 fish_label 候选 ----
        # 在这个阶段，并且帧号不超过 N_INIT_FRAMES，直接用 track_id 做显示 ID（1..9 等），
        # 这样前几帧不会出现 10~18 这种“跳号”，同时也保证一帧内不重复。
        if not has_perm and not has_stable_candidate and getattr(self, "frame_id", 0) <= getattr(self, "N_INIT_FRAMES", 0):
            for t in self.tracked_stracks:
                t.display_id = int(t.track_id)
            return

        # 每条鱼最多挑一个代表显示为 fish_id = fish_label+1
        for k, cand_list in candidates_by_fish.items():
            fish_id = k + 1
            if fish_id in used_ids:
                continue
            if not cand_list:
                continue

            cand_list.sort(key=lambda x: x[0], reverse=True)
            best_track = cand_list[0][1]
            best_track.display_id = fish_id
            used_ids.add(fish_id)

        # 3) 其余轨迹统一回退到基于 track_id 的唯一 ID，必要时加偏移避免和 1..9 冲突
        for t in self.tracked_stracks:
            if getattr(t, "display_id", None) is None:
                base = int(t.track_id)
                did = base
                while did in used_ids:
                    did += self.MAX_FISH
                t.display_id = did
                used_ids.add(did)

    def _update_permanent_tracks(self):
        """
        S4（非保守版）：
        - 对满足稳定条件（hits >= stable_frames，投票支持度 >= min_votes）的轨迹，
          使用 fish_label+1 作为候选 perm_id。
        - 如果该 perm_id 当前还没有代表，则直接升格为正式轨迹。
        - 如果已有代表 existing：
            * 若 existing 已经不是 Tracked 状态，或者太久未更新
              （frame_id - existing.end_frame > max_time_lost），
              或者当前 track 的 hits 更多，则由当前 track 接管 perm_id，
              旧代表降级为临时轨迹。
        这一版会在同一条鱼出现更稳定的新轨迹时允许“接管”，
        而不是像保守版那样永远不抢 perm_id。
        """
        for track in self.tracked_stracks:
            # 必须有 fish_label，且为有效标签
            if not hasattr(track, "fish_label"):
                continue
            if track.fish_label is None or track.fish_label < 0:
                continue

            # 统计当前 fish_label 在投票中的支持度
            support = 0
            if hasattr(track, "label_votes"):
                try:
                    ctr = Counter(track.label_votes)
                    support = ctr.get(track.fish_label, 0)
                except Exception:
                    # 兜底，防止 label_votes 不是 deque 时报错
                    try:
                        support = list(track.label_votes).count(track.fish_label)
                    except Exception:
                        support = 0

            # 轨迹还不够稳定：命中次数不足或投票支持度不足，跳过
            if track.hits < self.stable_frames or support < self.min_votes:
                continue

            # perm_id = fish_label + 1，限定在 [1, MAX_FISH] 范围内
            candidate_perm_id = int(track.fish_label) + 1
            if candidate_perm_id < 1 or candidate_perm_id > self.MAX_FISH:
                continue

            existing = self.permanent_tracks.get(candidate_perm_id)

            # 如果当前轨迹已经是“正式轨迹”，只需保证字典里有它即可
            if not track.is_temporary:
                if existing is None or existing is track:
                    self.permanent_tracks[candidate_perm_id] = track
                    self.used_perm_ids.add(candidate_perm_id)
                continue

            # 如果该 perm_id 还没有代表，直接升格当前轨迹
            if existing is None:
                track.promote_to_permanent(candidate_perm_id)
                self.permanent_tracks[candidate_perm_id] = track
                self.used_perm_ids.add(candidate_perm_id)
                continue

            # 判断旧代表是否应该被接管
            too_old = (
                (self.frame_id - existing.end_frame) > self.max_time_lost
                if hasattr(self, "max_time_lost") else False
            )

            margin = getattr(self, "takeover_hits_margin", 12)        ####
            old_hits = getattr(existing, "hits", 0)
            new_hits = getattr(track, "hits", 0)
            better_hits = new_hits >= (old_hits + margin)

            # 条件：旧代表状态不好 / 过期 / 新轨迹 hits 更高（含 margin），就让新轨迹接管 perm_id
            if existing.state != TrackState.Tracked or too_old or better_hits:
                # 旧代表降级为临时轨迹，收回 perm_id
                existing.is_temporary = True
                existing.perm_id = None

                # 当前轨迹升格为该 perm_id 的正式代表
                track.promote_to_permanent(candidate_perm_id)
                self.permanent_tracks[candidate_perm_id] = track
                self.used_perm_ids.add(candidate_perm_id)