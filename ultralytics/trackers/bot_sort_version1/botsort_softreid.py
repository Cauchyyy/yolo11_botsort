# ultralytics/trackers/botsort_softreid.py
# BoTSORT Soft-ReID variant — Kalman预测 + 匈牙利匹配（scipy）
from typing import List, Optional, Tuple, Callable
import numpy as np
import math, cv2, os, logging
from collections import deque, Counter

EPS = 1e-9
logger = logging.getLogger("botsort_softreid")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# -------- helpers --------
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
        return np.empty((0,0), dtype=float)
    a_n = l2_norm(a)
    b_n = l2_norm(b)
    sim = np.dot(a_n, b_n.T)
    return 1.0 - sim

def bbox_center(b):
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

def center_distance(a,b):
    ax,ay = bbox_center(a); bx,by = bbox_center(b)
    return math.hypot(ax-bx, ay-by)

def bbox_iou(a,b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
    inter = iw*ih
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + EPS
    return inter/union if union>0 else 0.0

def cxcywh_to_xyxy(cxcywh):
    cx,cy,w,h = cxcywh
    x1 = cx - 0.5*w; y1 = cy - 0.5*h; x2 = cx + 0.5*w; y2 = cy + 0.5*h
    return np.array([x1,y1,x2,y2], dtype=float)

def xyxy_to_cxcywh(xy):
    x1,y1,x2,y2 = xy
    w = max(1.0, x2-x1); h = max(1.0, y2-y1)
    cx = x1 + 0.5*w; cy = y1 + 0.5*h
    return np.array([cx,cy,w,h], dtype=float)

# -------- try imports: Kalman / Hungarian --------
_KF_CLASS = None
try:
    # try common project paths first
    from ultralytics.trackers.kalman import KalmanFilterXYWH  # type: ignore
    _KF_CLASS = KalmanFilterXYWH
except Exception:
    try:
        from ultralytics.kalman import KalmanFilterXYWH  # fallback path
        _KF_CLASS = KalmanFilterXYWH
    except Exception:
        _KF_CLASS = None

# fallback simple kalman if none found
if _KF_CLASS is None:
    class _SimpleKalman:
        """Very small Kalman-like placeholder: state [cx,cy,w,h,vx,vy,vw,vh]"""
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
                motion[i, 4+i] = 1.0
            mean = motion.dot(mean)
            cov = motion.dot(cov).dot(motion.T) + np.eye(8)*1e-2
            return mean, cov
        def update(self, mean, cov, meas):
            # simple direct replace for robustness
            mean[:4] = meas
            cov = cov * 0.9
            return mean, cov
        def project(self, mean, cov):
            # CORRECT: project state to measurement space: take cx,cy,w,h
            meas = mean[:4].copy()
            return meas, cov[:4, :4].copy()
    _KF_CLASS = _SimpleKalman

# Hungarian assignment: require scipy (no greedy fallback)
try:
    from scipy.optimize import linear_sum_assignment as _lsa
    def hungarian_assign(cost_mat, thresh=float("inf")):
        if cost_mat.size == 0:
            return [], list(range(cost_mat.shape[0])), list(range(cost_mat.shape[1]))
        # use linear_sum_assignment on a copy to be safe
        cm = cost_mat.copy()
        r, c = _lsa(cm)
        matches = []
        for i, j in zip(r, c):
            if cm[i, j] <= thresh:
                matches.append((int(i), int(j)))
        unmatched_a = [i for i in range(cost_mat.shape[0]) if i not in [m[0] for m in matches]]
        unmatched_b = [j for j in range(cost_mat.shape[1]) if j not in [m[1] for m in matches]]
        return matches, unmatched_a, unmatched_b
except Exception as e:
    raise ImportError("scipy is required for Hungarian assignment. Please install scipy (`pip install scipy`). Error: %s" % (e,))

# -------- Track class --------
class Track:
    def __init__(self, bbox: List[float], feat: Optional[np.ndarray], score: float, track_id: int,
                 frame_id: int, kf_obj, tracklet_len=10, vote_history=15, is_temporary=True, emb_dim: Optional[int]=None):
        self.bbox = [float(x) for x in bbox]
        self.score = float(score)
        self.track_id = int(track_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_frame = frame_id
        self.state = 'Tracked'
        self.is_temporary = bool(is_temporary)

        # embeddings
        self.tracklet = deque(maxlen=tracklet_len)
        self.emb_dim = emb_dim
        if feat is not None and getattr(feat, "size", 0):
            f = l2_norm(np.asarray(feat, dtype=np.float32))
            self.tracklet.append(f)
            self.feat_long = f.copy()
            # if emb_dim is None, infer from feat
            if self.emb_dim is None:
                self.emb_dim = f.shape[0]
        else:
            self.feat_long = None
        self.tracklet_len = int(tracklet_len)

        # label votes
        self.fish_label = -1
        self.label_votes = deque(maxlen=vote_history)
        self.label_conf = deque(maxlen=vote_history)

        # kf state
        self.kf = kf_obj
        cxcywh = xyxy_to_cxcywh(self.bbox)
        try:
            self.mean, self.cov = self.kf.initiate(cxcywh)
        except Exception:
            self.mean = np.r_[cxcywh, np.zeros_like(cxcywh)]
            self.cov = np.eye(8)

    def predict(self):
        try:
            self.mean, self.cov = self.kf.predict(self.mean, self.cov)
            # try to project to bbox
            try:
                meas, _ = self.kf.project(self.mean, self.cov)
                if meas.size >= 4:
                    self.bbox = cxcywh_to_xyxy(meas).tolist()
            except Exception:
                if self.mean.size >= 6:
                    est = cxcywh_to_xyxy(self.mean[[0,1,4,5]])
                    self.bbox = est.tolist()
        except Exception:
            logger.debug("KF predict failed for track %s", self.track_id)
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
        # update embedding
        if feat is not None and getattr(feat, "size", 0):
            f = l2_norm(np.asarray(feat, dtype=np.float32))
            self.tracklet.append(f)
            if self.feat_long is None:
                self.feat_long = f
                if self.emb_dim is None:
                    self.emb_dim = f.shape[0]
            else:
                self.feat_long = l2_norm((1.0-ema)*self.feat_long + ema * f)
        # update KF
        try:
            meas = xyxy_to_cxcywh(self.bbox)
            self.mean, self.cov = self.kf.update(self.mean, self.cov, meas)
        except Exception:
            logger.debug("KF update failed for track %s", self.track_id)

    def mark_lost(self):
        self.state = 'Lost'
        self.time_since_update += 1

    def get_tracklet_embedding(self):
        # return vector (emb_dim,) always (or fallback to 1-d zero if no info)
        if len(self.tracklet) == 0:
            if self.feat_long is not None:
                return self.feat_long
            if self.emb_dim is not None:
                return np.zeros((self.emb_dim,), dtype=float)
            return np.zeros((1,), dtype=float)
        arr = np.vstack(list(self.tracklet))
        avg = np.mean(arr, axis=0)
        return l2_norm(avg)

    def add_label_vote(self, label:int, conf:float):
        if label is None or int(label) < 0:
            return
        self.label_votes.append(int(label))
        self.label_conf.append(float(conf))
        ctr = Counter(self.label_votes)
        if len(ctr)>0:
            lbl, cnt = ctr.most_common(1)[0]
            self.fish_label = int(lbl)

    def label_confidence(self):
        return float(max(self.label_conf)) if len(self.label_conf)>0 else 0.0

# -------- Tracker class --------
class BoTSORTSoftReID:
    def __init__(self, encoder: Optional[Callable]=None,
                 w_app: float=0.35, w_motion: float=0.55, w_iou: float=0.10,
                 reid_reactivate_thresh: float=0.45, motion_thresh: float=200.0,
                 reserve_frames: int=150, max_age: int=400,
                 allow_new_tracks: bool=True,
                 stable_frames: int=5,
                 tracklet_len: int=10,
                 C_cls: float=0.60,
                 C_emb: float=0.65,
                 diagnostics_csv: Optional[str]=None,
                 logger_level:int=logging.INFO):
        """
        参数尽量与原版兼容（见你的run脚本）
        diagnostics_csv 参数保留为接口兼容，但该实现不再写CSV（按要求移除）
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

        # tracks
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []

        self._next_id = 1
        self.frame_id = 0

        # embedding dim (inferred on first encoder/fallback computation)
        self.emb_dim: Optional[int] = None

        # Kalman class instance
        try:
            self.kf = _KF_CLASS()
        except Exception:
            self.kf = _KF_CLASS()

        # diagnostics CSV removed per request; keep param for compat
        self.diagnostics_csv = None

        logger.setLevel(logger_level)

    # ----------------- core helpers -----------------
    def _get_reid_predictions(self, frame, bboxes):
        """
        Return: det_feats (NxD), pred_labels (N), pred_confs (N)
        Tries encoder(frame, rects) and optional encoder.predict_labels; falls back to simple HSV hist if no encoder.
        """
        N = len(bboxes)
        if N==0:
            # return empty array with known emb_dim if available, else 1
            dim = self.emb_dim if self.emb_dim is not None else 1
            return np.zeros((0, dim), dtype=np.float32), [], []

        H,W = frame.shape[:2]
        rects = []
        for b in bboxes:
            x1,y1,x2,y2 = int(max(0,b[0])), int(max(0,b[1])), int(min(W-1,b[2])), int(min(H-1,b[3]))
            if x2<=x1: x2 = min(W-1,x1+1)
            if y2<=y1: y2 = min(H-1,y1+1)
            rects.append([x1,y1,x2,y2])

        feats = None
        if self.encoder is not None:
            try:
                feats = self.encoder(frame, rects)
                feats = np.asarray(feats, dtype=np.float32)
            except Exception as e:
                logger.debug("encoder(frame,rects) failed: %s", e)
                feats = None

        if feats is None or feats.size == 0:
            # fallback: HSV hist (compact)
            tmp=[]
            for (x1,y1,x2,y2) in rects:
                patch = frame[y1:y2, x1:x2]
                if patch.size==0:
                    tmp.append(np.zeros((96,),dtype=np.float32)); continue
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                hst=[]
                for ch in range(3):
                    hh = cv2.calcHist([hsv],[ch],None,[32],[0,256]).flatten()
                    s = hh.sum()
                    if s>0: hh = hh/s
                    hst.append(hh)
                tmp.append(np.concatenate(hst).astype(np.float32))
            feats = np.vstack(tmp).astype(np.float32)

        feats = l2_norm(feats)

        # set emb_dim if not yet set
        if self.emb_dim is None and feats.ndim == 2 and feats.shape[1] > 0:
            self.emb_dim = feats.shape[1]

        # optional classifier labels from encoder
        pred_labels = [-1]*N
        pred_confs = [0.0]*N
        try:
            if self.encoder is not None and hasattr(self.encoder, "predict_labels"):
                labs, confs = self.encoder.predict_labels(frame, rects)
                labs = list(map(int, np.array(labs).tolist()))
                confs = list(map(float, np.array(confs).tolist()))
                if len(labs)==N: pred_labels = labs
                if len(confs)==N: pred_confs = confs
        except Exception:
            logger.debug("encoder.predict_labels failed")

        return feats, pred_labels, pred_confs

    def _predict_all(self):
        for tr in self.tracks + self.lost_tracks:
            tr.predict()

    def _compute_cost_matrix(self, tracks: List[Track], dets: List[List[float]], det_feats: np.ndarray):
        M = len(tracks); N = len(dets)
        if M==0 or N==0:
            return np.empty((M,N), dtype=float)
        track_embs = np.vstack([tr.get_tracklet_embedding() for tr in tracks])
        app_cost = cosine_distance_matrix(track_embs, det_feats) if track_embs.size and det_feats.size else np.ones((M,N), dtype=float)
        motion_cost = np.zeros((M,N), dtype=float)
        iou_cost = np.zeros((M,N), dtype=float)
        for i,tr in enumerate(tracks):
            for j,db in enumerate(dets):
                d = center_distance(tr.bbox, db)
                motion_cost[i,j] = min(1.0, float(d)/(self.motion_thresh + EPS))
                iou_cost[i,j] = 1.0 - bbox_iou(tr.bbox, db)
        cost = self.w_app * app_cost + self.w_motion * motion_cost + self.w_iou * iou_cost
        return cost

    # ----------------- main update -----------------
    def update(self, results, frame):
        """
        results: Nx6 array (x1,y1,x2,y2,score,cls)
        frame: BGR image
        returns: np.asarray list of outputs: [x1,y1,x2,y2, display_id, score, fish_label]
        """
        self.frame_id += 1

        if isinstance(results, np.ndarray):
            if results.size==0:
                dets=[]; scores=[]
            else:
                dets = [list(r[:4]) for r in results]
                scores = [float(r[4]) for r in results]
        else:
            dets = [list(r[:4]) for r in results]
            scores = [float(r[4]) for r in results]

        # no detections: just predict and output existing tracked
        if len(dets)==0:
            self._predict_all()
            outs=[]
            for tr in self.tracks:
                if tr.state=='Tracked':
                    display = (tr.fish_label+1) if tr.fish_label>=0 else tr.track_id
                    outs.append([float(tr.bbox[0]), float(tr.bbox[1]), float(tr.bbox[2]), float(tr.bbox[3]), int(display), float(tr.score), int(tr.fish_label)])
            return np.asarray(outs, dtype=object)

        # get REID feats + optional label preds
        det_feats, pred_labels, pred_confs = self._get_reid_predictions(frame, dets)
        if det_feats.ndim==1:
            det_feats = det_feats.reshape(1,-1)

        # 1) predict existing tracks
        self._predict_all()

        # 2) label-priority direct assignment (classifier strong vote)
        matched_det = set()
        for di,(lab,conf) in enumerate(zip(pred_labels, pred_confs)):
            if lab<0 or conf < self.C_cls: continue
            # try match active tracked with same label
            found=None
            for tr in self.tracks:
                if tr.fish_label==lab and tr.state=='Tracked':
                    found=tr; break
            if found and center_distance(found.bbox, dets[di]) <= self.motion_thresh*1.5:
                found.update_on_match(dets[di], det_feats[di] if det_feats.size else None, scores[di], self.frame_id)
                found.add_label_vote(lab, conf)
                matched_det.add(di)
                continue
            # try lost tracks with same label
            for lost in list(self.lost_tracks):
                if lost.fish_label==lab and (self.frame_id - lost.last_frame) <= (self.reserve_frames + self.max_age):
                    if center_distance(lost.bbox, dets[di]) <= self.motion_thresh*1.8:
                        lost.update_on_match(dets[di], det_feats[di] if det_feats.size else None, scores[di], self.frame_id)
                        try: self.lost_tracks.remove(lost)
                        except: pass
                        self.tracks.append(lost)
                        lost.add_label_vote(lab, conf)
                        matched_det.add(di)
                        break

        # 3) match active tracked <-> remaining detections via Hungarian on combined cost
        active_tracks = [tr for tr in self.tracks if tr.state=='Tracked']
        unmatched_det_indices = [i for i in range(len(dets)) if i not in matched_det]

        matches = []
        unmatched_tracks_idx = list(range(len(active_tracks)))
        unmatched_dets_idx = unmatched_det_indices.copy()

        if len(active_tracks)>0 and len(unmatched_dets_idx)>0:
            sub_dets = [dets[i] for i in unmatched_dets_idx]
            if det_feats.size:
                sub_feats = det_feats[unmatched_dets_idx]
            else:
                # create zero-features with emb_dim if known, else 1
                dim = self.emb_dim if self.emb_dim is not None else 1
                sub_feats = np.zeros((len(unmatched_dets_idx), dim), dtype=float)
            cost = self._compute_cost_matrix(active_tracks, sub_dets, sub_feats)
            # use hungarian (scipy)
            cost_copy = cost.copy()
            thresh = 1.0  # reject high-cost assignments (tunable)
            raw_matches, u_a, u_b = hungarian_assign(cost_copy, thresh=thresh)
            # translate matched indices to global det indices
            matches = [(int(a), int(unmatched_dets_idx[b])) for (a,b) in raw_matches]
            unmatched_tracks_idx = [i for i in range(len(active_tracks)) if i not in [m[0] for m in raw_matches]]
            unmatched_dets_idx = [unmatched_dets_idx[j] for j in u_b]

        # 4) apply matches -> update tracks
        for tr_idx, det_idx in matches:
            tr = active_tracks[tr_idx]
            tr.update_on_match(dets[det_idx], det_feats[det_idx] if det_feats.size else None, scores[det_idx], self.frame_id)
            pl = pred_labels[det_idx] if det_idx < len(pred_labels) else -1
            pc = pred_confs[det_idx] if det_idx < len(pred_confs) else 0.0
            if pl>=0:
                tr.add_label_vote(pl, pc)

        # 5) reactivate lost tracks by appearance matching
        if len(self.lost_tracks)>0 and len(unmatched_dets_idx)>0 and det_feats.size!=0:
            lost_list = list(self.lost_tracks)
            lost_embs = np.vstack([tr.get_tracklet_embedding() for tr in lost_list])
            sub_feats = det_feats[unmatched_dets_idx]
            if lost_embs.size and sub_feats.size:
                cost_lost = cosine_distance_matrix(lost_embs, sub_feats)
                cost_lost_copy = cost_lost.copy()
                thresh_lost = 1.0 - self.C_emb  # smaller means more similar
                used_lost=set(); used_det=set()
                while True:
                    if cost_lost_copy.size==0:
                        break
                    idx = np.unravel_index(np.argmin(cost_lost_copy), cost_lost_copy.shape)
                    li, cj = int(idx[0]), int(idx[1])
                    if cost_lost_copy[li,cj] >= 1e8:
                        break
                    val = float(cost_lost_copy[li,cj])
                    det_global_idx = unmatched_dets_idx[cj]
                    if val <= thresh_lost and center_distance(lost_list[li].bbox, dets[det_global_idx]) <= self.motion_thresh*2.0:
                        lost_tr = lost_list[li]
                        lost_tr.update_on_match(dets[det_global_idx], det_feats[det_global_idx] if det_feats.size else None, scores[det_global_idx], self.frame_id)
                        try: self.lost_tracks.remove(lost_tr)
                        except: pass
                        self.tracks.append(lost_tr)
                        used_lost.add(li); used_det.add(cj)
                    cost_lost_copy[li,:] = 1e9
                    cost_lost_copy[:,cj] = 1e9
                reactivated = set([unmatched_dets_idx[c] for c in used_det])
                unmatched_dets_idx = [d for d in unmatched_dets_idx if d not in reactivated]

        # 6) fallback center-distance matching for unmatched active tracks
        if len(unmatched_tracks_idx)>0 and len(unmatched_dets_idx)>0:
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

        # 7) remaining unmatched detections -> create temporary tracks (if allowed)
        currently_matched = set([m[1] for m in matches]) | matched_det
        remaining = [i for i in range(len(dets)) if i not in currently_matched]
        if len(remaining)>0 and self.allow_new_tracks:
            for d_idx in remaining:
                feat = det_feats[d_idx] if det_feats.size else None
                tr = Track(dets[d_idx], feat, scores[d_idx], self._next_id, self.frame_id, kf_obj=self.kf, tracklet_len=self.tracklet_len, is_temporary=True, emb_dim=self.emb_dim)
                self._next_id += 1
                self.tracks.append(tr)

        # 8) convert temporary tracks to permanent if stable by votes/frames
        for tr in list(self.tracks):
            if tr.is_temporary:
                lbl_counts = Counter(tr.label_votes)
                lbl, cnt = (-1,0) if len(lbl_counts)==0 else lbl_counts.most_common(1)[0]
                if tr.hits >= self.stable_frames and cnt >= max(2, int(self.stable_frames/2)):
                    if lbl >= 0:
                        tr.fish_label = int(lbl)
                        tr.is_temporary = False

        # 9) move tracks not updated -> lost; prune old lost
        to_move=[]
        for tr in list(self.tracks):
            if tr.time_since_update > 0:
                tr.mark_lost()
                to_move.append(tr)
        for tr in to_move:
            try:
                self.tracks.remove(tr)
                self.lost_tracks.append(tr)
            except Exception:
                pass

        to_delete = [tr for tr in self.lost_tracks if (self.frame_id - tr.last_frame) > (self.reserve_frames + self.max_age)]
        for tr in to_delete:
            try:
                self.lost_tracks.remove(tr)
                tr.state = 'Removed'
                self.removed_tracks.append(tr)
            except Exception:
                pass

        # assemble outputs
        outs=[]
        for tr in self.tracks:
            if tr.state == 'Tracked':
                display = (tr.fish_label+1) if tr.fish_label>=0 else tr.track_id
                outs.append([float(tr.bbox[0]), float(tr.bbox[1]), float(tr.bbox[2]), float(tr.bbox[3]), int(display), float(tr.score), int(tr.fish_label)])
        return np.asarray(outs, dtype=object)

    def reset(self):
        self.tracks=[]; self.lost_tracks=[]; self.removed_tracks=[]
        self._next_id=1; self.frame_id=0
