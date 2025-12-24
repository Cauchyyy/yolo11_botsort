# ultralytics/trackers/botsort_softreid.py
# BoTSORT Soft-ReID variant — Kalman prediction + Hungarian matching
# - ONLY use ReID encoder for appearance (no HSV fallback)
# - Keep main BoTSORT logic, but accept top_k to only track highest-confidence detections

from typing import List, Optional, Callable
import numpy as np
import math, cv2, logging
from collections import deque, Counter

EPS = 1e-9
logger = logging.getLogger("botsort_softreid")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def l2_norm(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if x.ndim == 1:
        n = np.linalg.norm(x) + EPS
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + EPS
    return x / n


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray):
    if a is None or b is None or a.size == 0 or b.size == 0:
        return np.empty((0, 0), dtype=float)
    a_n = l2_norm(a); b_n = l2_norm(b)
    sim = np.dot(a_n, b_n.T)
    return 1.0 - sim


def bbox_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def center_distance(a, b):
    ax, ay = bbox_center(a); bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def cxcywh_to_xyxy(cxcywh):
    cx, cy, w, h = cxcywh
    x1 = cx - 0.5 * w; y1 = cy - 0.5 * h; x2 = cx + 0.5 * w; y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=float)


def xyxy_to_cxcywh(xy):
    x1, y1, x2, y2 = xy
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w; cy = y1 + 0.5 * h
    return np.array([cx, cy, w, h], dtype=float)


# Try import Kalman; fallback to simple one
_KF_CLASS = None
try:
    from ultralytics.trackers.kalman import KalmanFilterXYWH  # type: ignore
    _KF_CLASS = KalmanFilterXYWH
except Exception:
    try:
        from ultralytics.kalman import KalmanFilterXYWH
        _KF_CLASS = KalmanFilterXYWH
    except Exception:
        _KF_CLASS = None

if _KF_CLASS is None:
    class _SimpleKalman:
        def __init__(self):
            self.ndim = 4
        def initiate(self, meas):
            mean_pos = np.asarray(meas, dtype=float)
            mean = np.r_[mean_pos, np.zeros_like(mean_pos)]
            cov = np.eye(8) * 1.0
            return mean, cov
        def predict(self, mean, cov):
            motion = np.eye(8)
            for i in range(4):
                motion[i, 4 + i] = 1.0
            mean = motion.dot(mean)
            cov = motion.dot(cov).dot(motion.T) + np.eye(8) * 1e-2
            return mean, cov
        def update(self, mean, cov, meas):
            mean[:4] = meas
            cov = cov * 0.9
            return mean, cov
        def project(self, mean, cov):
            # return measurement as [cx,cy,w,h]
            meas = mean[[0, 1, 2, 3]]
            return meas, cov[:4, :4]
    _KF_CLASS = _SimpleKalman


# Hungarian assignment: prefer scipy, else greedy fallback
try:
    from scipy.optimize import linear_sum_assignment as _lsa
    def hungarian_assign(cost_mat, thresh=float("inf")):
        if cost_mat.size == 0:
            return [], list(range(cost_mat.shape[0])), list(range(cost_mat.shape[1]))
        r, c = _lsa(cost_mat)
        matches = []
        for i, j in zip(r, c):
            if cost_mat[i, j] <= thresh:
                matches.append((int(i), int(j)))
        unmatched_a = [i for i in range(cost_mat.shape[0]) if i not in [m[0] for m in matches]]
        unmatched_b = [j for j in range(cost_mat.shape[1]) if j not in [m[1] for m in matches]]
        return matches, unmatched_a, unmatched_b
except Exception:
    def hungarian_assign(cost_mat, thresh=float("inf")):
        if cost_mat.size == 0:
            return [], list(range(cost_mat.shape[0])), list(range(cost_mat.shape[1]))
        N, M = cost_mat.shape
        matches = []; used_r = set(); used_c = set()
        while True:
            idx = np.unravel_index(np.argmin(cost_mat), cost_mat.shape)
            r, c = int(idx[0]), int(idx[1])
            if cost_mat[r, c] >= thresh:
                break
            if r in used_r or c in used_c:
                cost_mat[r, c] = 1e9
                continue
            matches.append((r, c))
            used_r.add(r); used_c.add(c)
            cost_mat[r, :] = 1e9; cost_mat[:, c] = 1e9
            if len(used_r) == N or len(used_c) == M:
                break
        unmatched_a = [i for i in range(N) if i not in used_r]
        unmatched_b = [j for j in range(M) if j not in used_c]
        return matches, unmatched_a, unmatched_b


class Track:
    def __init__(self, bbox: List[float], feat: Optional[np.ndarray], score: float, track_id: int,
                 frame_id: int, kf_obj, tracklet_len=10, vote_history=15, is_temporary=True, feat_dim: Optional[int]=None):
        self.bbox = [float(x) for x in bbox]
        self.score = float(score)
        self.track_id = int(track_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_frame = frame_id
        self.state = 'Tracked'
        self.is_temporary = bool(is_temporary)

        self.tracklet = deque(maxlen=tracklet_len)
        if feat is not None and getattr(feat, "size", 0):
            f = l2_norm(np.asarray(feat, dtype=np.float32))
            self.tracklet.append(f)
            self.feat_long = l2_norm(f).copy()
        else:
            self.feat_long = None
        self.tracklet_len = int(tracklet_len)

        self.fish_label = -1
        self.label_votes = deque(maxlen=vote_history)
        self.label_conf = deque(maxlen=vote_history)

        self.kf = kf_obj
        cxcywh = xyxy_to_cxcywh(self.bbox)
        try:
            self.mean, self.cov = self.kf.initiate(cxcywh)
        except Exception:
            self.mean = np.r_[cxcywh, np.zeros_like(cxcywh)]
            self.cov = np.eye(8)

        # Ensure fallback embedding dimension is stored for consistent returns
        self.feat_dim = int(feat_dim) if feat_dim is not None else (self.feat_long.shape[0] if (self.feat_long is not None and getattr(self.feat_long, 'shape', None)) else None)

    def predict(self):
        try:
            self.mean, self.cov = self.kf.predict(self.mean, self.cov)
            try:
                meas, _ = self.kf.project(self.mean, self.cov)
                if meas.size >= 4:
                    self.bbox = cxcywh_to_xyxy(meas).tolist()
            except Exception:
                if self.mean.size >= 6:
                    est = cxcywh_to_xyxy(self.mean[[0, 1, 4, 5]])
                    self.bbox = est.tolist()
        except Exception:
            pass
        self.age += 1
        self.time_since_update += 1

    def update_on_match(self, bbox, feat, score, frame_id, ema=0.06):
        old_cx, old_cy = bbox_center(self.bbox)
        new_cx, new_cy = bbox_center(bbox)
        self.vx = new_cx - old_cx
        self.vy = new_cy - old_cy
        self.bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        self.score = float(score)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_frame = frame_id
        self.state = 'Tracked'
        if feat is not None and getattr(feat, "size", 0):
            f = l2_norm(np.asarray(feat, dtype=np.float32))
            self.tracklet.append(f)
            if self.feat_long is None:
                self.feat_long = f
            else:
                self.feat_long = l2_norm((1.0 - ema) * self.feat_long + ema * f)
        try:
            meas = xyxy_to_cxcywh(self.bbox)
            self.mean, self.cov = self.kf.update(self.mean, self.cov, meas)
        except Exception:
            pass

    def mark_lost(self):
        self.state = 'Lost'
        self.time_since_update += 1

    def get_tracklet_embedding(self):
        if len(self.tracklet) == 0:
            if self.feat_long is None:
                if self.feat_dim is not None:
                    return np.zeros((self.feat_dim,), dtype=np.float32)
                else:
                    return np.zeros((1,), dtype=np.float32)
            return self.feat_long
        arr = np.vstack(list(self.tracklet))
        avg = np.mean(arr, axis=0)
        return l2_norm(avg)

    def add_label_vote(self, label: int, conf: float):
        if label is None or int(label) < 0:
            return
        self.label_votes.append(int(label))
        self.label_conf.append(float(conf))
        ctr = Counter(self.label_votes)
        if len(ctr) > 0:
            lbl, cnt = ctr.most_common(1)[0]
            self.fish_label = int(lbl)


class BoTSORTSoftReID:
    def __init__(self, encoder: Optional[Callable] = None,
                 w_app: float = 0.35, w_motion: float = 0.55, w_iou: float = 0.10,
                 reid_reactivate_thresh: float = 0.45, motion_thresh: float = 200.0,
                 reserve_frames: int = 150, max_age: int = 400,
                 allow_new_tracks: bool = True,
                 stable_frames: int = 5,
                 tracklet_len: int = 10,
                 C_cls: float = 0.60,
                 C_emb: float = 0.65,
                 logger_level: int = logging.INFO,
                 match_thresh: float = 0.6,
                 move_to_lost_thresh: int = 2,
                 top_k: int = 9):
        """
        key args:
         - encoder: callable(frame, rects) -> NxD embeddings (REQUIRED for meaningful ReID)
         - top_k: number of top detections (by confidence) to keep per frame (others discarded)
        """
        self.encoder = encoder
        self.w_app = float(w_app); self.w_motion = float(w_motion); self.w_iou = float(w_iou)
        self.reid_reactivate_thresh = float(reid_reactivate_thresh)
        self.motion_thresh = float(motion_thresh)
        self.reserve_frames = int(reserve_frames)
        self.max_age = int(max_age)
        self.allow_new_tracks = bool(allow_new_tracks)
        self.stable_frames = int(stable_frames)
        self.tracklet_len = int(tracklet_len)
        self.C_cls = float(C_cls)
        self.C_emb = float(C_emb)

        self.match_thresh = float(match_thresh)
        self.move_to_lost_thresh = int(move_to_lost_thresh)
        self.top_k = int(top_k)

        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []

        self._next_id = 1
        self.frame_id = 0

        try:
            self.kf = _KF_CLASS()
        except Exception:
            self.kf = _KF_CLASS()

        # Determine embedding dim: try encoder once, else fallback default
        self.feat_dim = None
        if self.encoder is not None:
            try:
                dummy_img = np.zeros((16, 16, 3), dtype=np.uint8)
                dummy_rects = [[0, 0, 1, 1]]
                feats_test = self.encoder(dummy_img, dummy_rects)
                feats_test = np.asarray(feats_test)
                if feats_test.ndim == 1:
                    self.feat_dim = int(feats_test.shape[0])
                else:
                    self.feat_dim = int(feats_test.shape[-1])
            except Exception:
                logger.debug("Could not infer encoder feat dim; defaulting to 256")
                self.feat_dim = 256
        else:
            # No encoder provided: still set a default dim for zero embeddings
            logger.warning("No ReID encoder provided: appearance matching will be inactive (zero embeddings).")
            self.feat_dim = 256

        logger.setLevel(logger_level)

    def _get_reid_predictions(self, frame, bboxes):
        """
        Return: det_feats (NxD), pred_labels (N), pred_confs (N)
        ONLY uses self.encoder. If encoder is None or fails, returns zero embeddings.
        """
        N = len(bboxes)
        if N == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32), [], []
        H, W = frame.shape[:2]
        rects = []
        for b in bboxes:
            x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(W - 1, b[2])), int(min(H - 1, b[3]))
            if x2 <= x1: x2 = min(W - 1, x1 + 1)
            if y2 <= y1: y2 = min(H - 1, y1 + 1)
            rects.append([x1, y1, x2, y2])

        feats = None
        if self.encoder is not None:
            try:
                feats = self.encoder(frame, rects)
                feats = np.asarray(feats, dtype=np.float32)
                if feats.ndim == 1 and N == 1:
                    feats = feats.reshape(1, -1)
            except Exception as e:
                logger.exception("ReID encoder failure: %s", e)
                feats = None

        if feats is None or feats.size == 0:
            # No HSV fallback anymore — return zero embeddings (appearance won't help)
            feats = np.zeros((N, self.feat_dim), dtype=np.float32)
            pred_labels = [-1] * N
            pred_confs = [0.0] * N
            return feats, pred_labels, pred_confs

        feats = l2_norm(feats)

        # optional classifier labels from encoder
        pred_labels = [-1] * N
        pred_confs = [0.0] * N
        try:
            if hasattr(self.encoder, "predict_labels"):
                labs, confs = self.encoder.predict_labels(frame, rects)
                labs = list(map(int, np.array(labs).tolist()))
                confs = list(map(float, np.array(confs).tolist()))
                if len(labs) == N: pred_labels = labs
                if len(confs) == N: pred_confs = confs
        except Exception:
            logger.debug("predict_labels not available or failed")

        return feats, pred_labels, pred_confs

    def _predict_all(self):
        for tr in self.tracks + self.lost_tracks:
            tr.predict()

    def _compute_cost_matrix(self, tracks: List[Track], dets: List[List[float]], det_feats: np.ndarray):
        M = len(tracks); N = len(dets)
        if M == 0 or N == 0:
            return np.empty((M, N), dtype=float)

        track_embs = np.vstack([tr.get_tracklet_embedding() for tr in tracks])
        app_cost = cosine_distance_matrix(track_embs, det_feats) if track_embs.size and det_feats.size else np.ones((M, N), dtype=float)

        t_centers = np.array([bbox_center(tr.bbox) for tr in tracks], dtype=np.float32)
        d_centers = np.array([bbox_center(db) for db in dets], dtype=np.float32)
        diff = t_centers[:, None, :] - d_centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        motion_cost = np.minimum(1.0, dists / (self.motion_thresh + EPS))

        t_boxes = np.array([tr.bbox for tr in tracks], dtype=np.float32)
        d_boxes = np.array(dets, dtype=np.float32)
        t_x1 = t_boxes[:, 0][:, None]; t_y1 = t_boxes[:, 1][:, None]; t_x2 = t_boxes[:, 2][:, None]; t_y2 = t_boxes[:, 3][:, None]
        d_x1 = d_boxes[:, 0][None, :]; d_y1 = d_boxes[:, 1][None, :]; d_x2 = d_boxes[:, 2][None, :]; d_y2 = d_boxes[:, 3][None, :]
        inter_x1 = np.maximum(t_x1, d_x1); inter_y1 = np.maximum(t_y1, d_y1)
        inter_x2 = np.minimum(t_x2, d_x2); inter_y2 = np.minimum(t_y2, d_y2)
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_t = np.maximum(0.0, (t_x2 - t_x1) * (t_y2 - t_y1))
        area_d = np.maximum(0.0, (d_x2 - d_x1) * (d_y2 - d_y1))
        union = area_t + area_d - inter + EPS
        iou = inter / union
        iou_cost = 1.0 - iou

        cost = self.w_app * app_cost + self.w_motion * motion_cost + self.w_iou * iou_cost
        return cost

    def update(self, results, frame):
        """
        results: Nx6 array (x1,y1,x2,y2,score,cls) OR list-like of same
        Note: only top_k detections (by score) are kept per frame; others are discarded.
        """
        self.frame_id += 1

        # parse detections -> dets (xyxy) and scores
        if isinstance(results, np.ndarray):
            if results.size == 0:
                dets = []; scores = []
            else:
                dets = [list(r[:4]) for r in results]; scores = [float(r[4]) for r in results]
        else:
            dets = [list(r[:4]) for r in results]; scores = [float(r[4]) for r in results]

        # keep only top_k detections by score (discard the rest)
        if len(dets) > 0 and self.top_k is not None and self.top_k > 0 and len(dets) > self.top_k:
            idxs = np.argsort(scores)[::-1][:self.top_k]  # top by score desc
            dets = [dets[i] for i in idxs]
            scores = [scores[i] for i in idxs]

        if len(dets) == 0:
            self._predict_all()
            outs = []
            for tr in self.tracks:
                if tr.state == 'Tracked':
                    display = (tr.fish_label + 1) if tr.fish_label >= 0 else tr.track_id
                    outs.append([float(tr.bbox[0]), float(tr.bbox[1]), float(tr.bbox[2]), float(tr.bbox[3]),
                                 int(display), float(tr.score), int(tr.fish_label)])
            return np.asarray(outs, dtype=object)

        det_feats, pred_labels, pred_confs = self._get_reid_predictions(frame, dets)
        if det_feats.ndim == 1:
            det_feats = det_feats.reshape(1, -1)

        # predict existing tracks first
        self._predict_all()

        # 1) label-priority direct assignment (classifier strong vote)
        matched_det = set()
        for di, (lab, conf) in enumerate(zip(pred_labels, pred_confs)):
            if lab < 0 or conf < self.C_cls:
                continue
            found = None
            for tr in self.tracks:
                if tr.fish_label == lab and tr.state == 'Tracked':
                    found = tr; break
            if found and center_distance(found.bbox, dets[di]) <= self.motion_thresh * 1.5:
                found.update_on_match(dets[di], det_feats[di] if det_feats.size else None, scores[di], self.frame_id)
                found.add_label_vote(lab, conf)
                matched_det.add(di); continue
            for lost in list(self.lost_tracks):
                if lost.fish_label == lab and (self.frame_id - lost.last_frame) <= (self.reserve_frames + self.max_age):
                    if center_distance(lost.bbox, dets[di]) <= self.motion_thresh * 1.8:
                        lost.update_on_match(dets[di], det_feats[di] if det_feats.size else None, scores[di], self.frame_id)
                        try: self.lost_tracks.remove(lost)
                        except: pass
                        self.tracks.append(lost)
                        lost.add_label_vote(lab, conf)
                        matched_det.add(di)
                        break

        # 2) Hungarian matching for remaining active tracks <-> remaining dets
        active_tracks = [tr for tr in self.tracks if tr.state == 'Tracked']
        unmatched_det_indices = [i for i in range(len(dets)) if i not in matched_det]

        matches = []; unmatched_tracks_idx = list(range(len(active_tracks))); unmatched_dets_idx = unmatched_det_indices.copy()

        if len(active_tracks) > 0 and len(unmatched_dets_idx) > 0:
            sub_dets = [dets[i] for i in unmatched_dets_idx]
            sub_feats = det_feats[unmatched_dets_idx] if det_feats.size else np.zeros((len(unmatched_dets_idx), det_feats.shape[1] if det_feats.size else 1))
            cost = self._compute_cost_matrix(active_tracks, sub_dets, sub_feats)
            # use absolute match_thresh on cost (cost values are in [0,1] components combined)
            raw_matches, u_a, u_b = hungarian_assign(cost.copy(), thresh=self.match_thresh)
            matches = [(int(a), int(unmatched_dets_idx[b])) for (a, b) in raw_matches]
            unmatched_tracks_idx = [i for i in range(len(active_tracks)) if i not in [m[0] for m in raw_matches]]
            unmatched_dets_idx = [unmatched_dets_idx[j] for j in u_b]

        # apply matches
        for tr_idx, det_idx in matches:
            tr = active_tracks[tr_idx]
            tr.update_on_match(dets[det_idx], det_feats[det_idx] if det_feats.size else None, scores[det_idx], self.frame_id)
            pl = pred_labels[det_idx] if det_idx < len(pred_labels) else -1
            pc = pred_confs[det_idx] if det_idx < len(pred_confs) else 0.0
            if pl >= 0:
                tr.add_label_vote(pl, pc)

        # re-activate lost_tracks by appearance matching (vectorized approach)
        if len(self.lost_tracks) > 0 and len(unmatched_dets_idx) > 0 and det_feats.size != 0:
            lost_list = list(self.lost_tracks)
            lost_embs = np.vstack([tr.get_tracklet_embedding() for tr in lost_list])
            sub_feats = det_feats[unmatched_dets_idx]
            if lost_embs.size and sub_feats.size:
                cost_lost = cosine_distance_matrix(lost_embs, sub_feats)  # L x U
                # use Hungarian on cost_lost with threshold
                raw_matches_lost, u_l, u_d = hungarian_assign(cost_lost.copy(), thresh=(1.0 - self.C_emb))
                used_det = set()
                for (li, dj) in raw_matches_lost:
                    det_global_idx = unmatched_dets_idx[int(dj)]
                    if center_distance(lost_list[int(li)].bbox, dets[det_global_idx]) <= self.motion_thresh * 2.0:
                        lost_tr = lost_list[int(li)]
                        lost_tr.update_on_match(dets[det_global_idx], det_feats[det_global_idx] if det_feats.size else None, scores[det_global_idx], self.frame_id)
                        try: self.lost_tracks.remove(lost_tr)
                        except: pass
                        self.tracks.append(lost_tr)
                        used_det.add(det_global_idx)
                unmatched_dets_idx = [d for d in unmatched_dets_idx if d not in used_det]

        # fallback center-distance matching for unmatched active tracks
        if len(unmatched_tracks_idx) > 0 and len(unmatched_dets_idx) > 0:
            for idx_tr in unmatched_tracks_idx:
                tr = active_tracks[idx_tr]
                best_j = -1; best_dist = 1e9
                for dj in unmatched_dets_idx:
                    dist = center_distance(tr.bbox, dets[dj])
                    if dist < best_dist and dist < (self.motion_thresh * 1.2):
                        best_dist = dist; best_j = dj
                if best_j >= 0:
                    tr.update_on_match(dets[best_j], det_feats[best_j] if det_feats.size else None, scores[best_j], self.frame_id)
                    if best_j in unmatched_dets_idx: unmatched_dets_idx.remove(best_j)

        # remaining unmatched detections -> create temporary tracks (if allowed)
        currently_matched = set([m[1] for m in matches]) | matched_det
        remaining = [i for i in range(len(dets)) if i not in currently_matched]
        if len(remaining) > 0 and self.allow_new_tracks:
            for d_idx in remaining:
                tr = Track(dets[d_idx], det_feats[d_idx] if det_feats.size else None, scores[d_idx],
                           self._next_id, self.frame_id, kf_obj=self.kf,
                           tracklet_len=self.tracklet_len, is_temporary=True, feat_dim=self.feat_dim)
                self._next_id += 1
                self.tracks.append(tr)

        # convert temporary tracks to permanent if stable by votes/frames
        for tr in list(self.tracks):
            if tr.is_temporary:
                lbl_counts = Counter(tr.label_votes)
                lbl, cnt = (-1, 0) if len(lbl_counts) == 0 else lbl_counts.most_common(1)[0]
                if tr.hits >= self.stable_frames and cnt >= max(2, int(self.stable_frames / 2)):
                    if lbl >= 0:
                        tr.fish_label = int(lbl); tr.is_temporary = False

        # move tracks not updated -> lost; prune old lost
        to_move = []
        for tr in list(self.tracks):
            if tr.time_since_update > self.move_to_lost_thresh:
                tr.mark_lost(); to_move.append(tr)
        for tr in to_move:
            try:
                self.tracks.remove(tr); self.lost_tracks.append(tr)
            except Exception:
                pass

        to_delete = [tr for tr in self.lost_tracks if (self.frame_id - tr.last_frame) > (self.reserve_frames + self.max_age)]
        for tr in to_delete:
            try:
                self.lost_tracks.remove(tr); tr.state = 'Removed'; self.removed_tracks.append(tr)
            except Exception:
                pass
        # optional: keep removed_tracks bounded to avoid unbounded memory growth
        if len(self.removed_tracks) > 10000:
            self.removed_tracks = self.removed_tracks[-5000:]

        # assemble outputs
        outs = []
        for tr in self.tracks:
            if tr.state == 'Tracked':
                display = (tr.fish_label + 1) if tr.fish_label >= 0 else tr.track_id
                outs.append([float(tr.bbox[0]), float(tr.bbox[1]), float(tr.bbox[2]), float(tr.bbox[3]),
                             int(display), float(tr.score), int(tr.fish_label)])
        return np.asarray(outs, dtype=object)

    def reset(self):
        self.tracks = []; self.lost_tracks = []; self.removed_tracks = []
        self._next_id = 1; self.frame_id = 0
