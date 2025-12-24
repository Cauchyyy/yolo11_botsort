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

    #def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
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
        #########################
        self.reid_conf = reid_conf
        self.cls_history = deque()
        self.label_votes = deque(maxlen=15)
        self.fish_label = -1
        self.hits = 0
        self.prev_center = None
        self.speed = 0.0
        ################################

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
    ##############################################################################################  
    def activate(self, kalman_filter, frame_id):
        """Activates a new tracklet and records its initial classification for later voting."""
        if self.cls is not None:
            self.cls_history.append(int(self.cls))
        super().activate(kalman_filter, frame_id)
        self.hits = 1
        self._update_speed()
    ############################################################################################

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        if new_track.cls is not None:
            self.cls_history.append(int(new_track.cls))
        super().re_activate(new_track, frame_id, new_id)
        self.hits += 1
        self._update_speed()

    def update(self, new_track, frame_id):
        """Updates the YOLOv8 instance with new track information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
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
        """Return tracking result using per-frame display_id or fish_label-based fallback."""
        coords = self.xyxy if self.angle is None else self.xywha

        display_id = getattr(self, "display_id", None)
        if display_id is None:
            if self.fish_label is not None and self.fish_label >= 0:
                display_id = int(self.fish_label) + 1
            else:
                display_id = int(self.track_id)
        else:
            display_id = int(display_id)

        perm = -1  # S1-only: perm_id is not used
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

        # S1: fish label voting configuration
        self.MAX_FISH = 9
        self.stable_frames = getattr(args, "stable_frames", 8)
        self.min_votes = getattr(args, "min_votes", 4)

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
        """Run one tracking step and assign per-frame unique display IDs based on S1 voting."""
        if img is not None:
            self.img_h, self.img_w = img.shape[:2]

        super().update(results, img)
        self._assign_display_ids()

        outputs = []
        for track in self.tracked_stracks:
            if track.is_activated:
                outputs.append(track.result)

        if len(outputs) == 0:
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
        """Calculates distances between tracks and detections using IoU and optionally ReID embeddings."""
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
    
    def _assign_display_ids(self):
        """Assign per-frame unique display IDs based solely on S1 voting results."""
        if not self.tracked_stracks:
            return

        for t in self.tracked_stracks:
            if hasattr(t, "display_id"):
                t.display_id = None

        used_ids = set()
        candidates_by_fish = {k: [] for k in range(self.MAX_FISH)}

        for t in self.tracked_stracks:
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
                try:
                    support = list(label_votes).count(fish_label)
                except Exception:
                    support = 0

            hits = getattr(t, "hits", 0)
            conf = getattr(t, "reid_conf", 0.0) or 0.0
            det_score = getattr(t, "score", 0.0) or 0.0

            if hits < self.stable_frames or support < self.min_votes:
                continue

            score = (hits, support, conf, det_score)
            candidates_by_fish[k].append((score, t))

        for k, cand_list in candidates_by_fish.items():
            fish_id = k + 1
            if not cand_list:
                continue

            cand_list.sort(key=lambda x: x[0], reverse=True)
            best_track = cand_list[0][1]
            if best_track.display_id is None:
                best_track.display_id = fish_id
                used_ids.add(fish_id)

        for t in self.tracked_stracks:
            if getattr(t, "display_id", None) is not None:
                continue
            base = int(t.track_id)
            did = base
            while did in used_ids:
                did += self.MAX_FISH
            t.display_id = did
            used_ids.add(did)
