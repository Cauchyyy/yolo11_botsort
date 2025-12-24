# ultralytics/trackers/botsort_softreid.py
# Soft-ReID BoTSORT variant:
# - REID used as soft evidence (prob/conf + embedding)
# - allow temporary new tracks and later convert to permanent by votes/tracklet stability
# - tracklet-level embedding aggregation
# - diagnostics CSV output
#
# Expectations:
# - encoder(frame, list_of_bboxes) -> np.ndarray (N, D) embeddings  (common)
# - encoder.predict_labels(frame, list_of_bboxes) -> (labels, confs) optional
#
# If predict_labels not available, code will fallback to embeddings-only matching.

from typing import List, Optional
import numpy as np, math, cv2, csv, os
from collections import deque, Counter

EPS = 1e-9

def l2_norm(x: np.ndarray):     ##正则化
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if x.ndim == 1:
        n = np.linalg.norm(x) + EPS
        return x / n
    else:
        n = np.linalg.norm(x, axis=1, keepdims=True) + EPS
        return x / n

def cosine_distance_matrix(a: np.ndarray, b: np.ndarray):    #计算余弦距离
    if a is None or b is None or a.size == 0 or b.size == 0:
        return np.empty((0,0), dtype=float)
    a_n = l2_norm(a)
    b_n = l2_norm(b)
    sim = np.dot(a_n, b_n.T)
    return 1.0 - sim  # smaller = more similar

def bbox_center(b):  #返回 bbox 的中心坐标 (cx, cy)
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def center_distance(a,b):  #返回两个 bbox 中心点的欧氏距离
    ax,ay = bbox_center(a); bx,by = bbox_center(b)
    return math.hypot(ax-bx, ay-by)

def bbox_iou(a,b):  #计算两个 bbox 的 IoU（交并比）
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
    inter = iw*ih
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter/union if union>0 else 0.0

class Track:  #单条轨迹的状态与管理）
    def __init__(self, bbox, feat, score, track_id, frame_id, is_temporary=True, tracklet_len=10, vote_history=15):
        self.bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        self.score = float(score)
        self.track_id = int(track_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_frame = frame_id
        self.state = 'Tracked'  # Tracked / Lost / Removed
        self.is_temporary = bool(is_temporary)

        # embedding storages
        self.feat_long = l2_norm(feat) if feat is not None and feat.size else None  # EMA/long-term 长期 EMA embedding（用于稳定长期相似度）
        self.tracklet = deque(maxlen=tracklet_len)  # recent embeddings  #最近 N 帧 embedding 的 deque（用于短期平均）
        if feat is not None and feat.size:
            self.tracklet.append(l2_norm(feat))
        self.tracklet_len = tracklet_len

        # label voting (if matched to classifier predictions)
        self.fish_label = -1  #若轨迹被归入某类别（或特定个体标签），保存在这里
        self.label_votes = deque(maxlen=vote_history)
        self.label_conf = deque(maxlen=vote_history)

        # simple motion velocity (pixel/frame)
        self.vx = 0.0; self.vy = 0.0

    def update_on_match(self, bbox, feat, score, frame_id, ema_momentum=0.06):  #当检测与某轨迹匹配时，用新信息更新轨迹
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
        if feat is not None and feat.size:
            f = l2_norm(feat)
            self.tracklet.append(f)
            if self.feat_long is None:
                self.feat_long = f
            else:
                self.feat_long = l2_norm((1.0-ema_momentum)*self.feat_long + ema_momentum * f)

    def mark_lost(self):
        self.time_since_update += 1
        self.state = 'Lost'

    def get_tracklet_embedding(self):
        if len(self.tracklet) == 0:
            return self.feat_long if self.feat_long is not None else np.zeros((1,), dtype=float)
        arr = np.vstack(list(self.tracklet))
        avg = np.mean(arr, axis=0)
        return l2_norm(avg)

    def add_label_vote(self, label, conf):  #将新的 classifier 投票加入 label_votes 与 label_conf，并用 Counter 更新 fish_label 为出现频次最高的标签（多数表决）。
        if label is None or int(label) < 0:
            return
        self.label_votes.append(int(label))
        self.label_conf.append(float(conf))
        ctr = Counter(self.label_votes)
        if len(ctr) > 0:
            lbl, cnt = ctr.most_common(1)[0]
            self.fish_label = int(lbl)

    def label_confidence(self):
        if len(self.label_conf) == 0:
            return 0.0
        return float(max(self.label_conf))

class BoTSORTSoftReID:
    def __init__(self, encoder: Optional[object]=None,
                 w_app: float=0.35, w_motion: float=0.55, w_iou: float=0.10,
                 reid_reactivate_thresh: float=0.45, motion_thresh: float=200.0,
                 reserve_frames: int=150, max_age: int=400,
                 allow_new_tracks: bool=True,
                 stable_frames: int=5,
                 tracklet_len: int=10,
                 C_cls: float=0.60,
                 C_emb: float=0.65,
                 diagnostics_csv: Optional[str]=None):
        """
        Use REID as soft evidence:
        - C_cls: classifier softmax threshold for direct assignment (0..1)
        - C_emb: embedding cosine similarity threshold for fallback (0..1)
        """
        self.encoder = encoder
        self.w_app = float(w_app)
        self.w_motion = float(w_motion)
        self.w_iou = float(w_iou)
        self.reid_reactivate_thresh = float(reid_reactivate_thresh)
        self.motion_thresh = float(motion_thresh)
        self.reserve_frames = int(reserve_frames)
        self.max_age = int(max_age)
        self.allow_new_tracks = bool(allow_new_tracks)
        self.stable_frames = int(stable_frames)
        self.tracklet_len = int(tracklet_len)

        self.C_cls = float(C_cls)
        self.C_emb = float(C_emb)

        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []

        self._next_id = 1
        self.frame_id = 0

        # diagnostics CSV
        self.diagnostics_csv = diagnostics_csv
        if diagnostics_csv:
            ddir = os.path.dirname(diagnostics_csv)
            if ddir and not os.path.exists(ddir):
                os.makedirs(ddir, exist_ok=True)
            with open(self.diagnostics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame","det_idx","x1","y1","x2","y2","score","pred_label","pred_conf","top3_track_ids","top3_track_dists","center_dist","iou","assigned_track","display_id"])

    # helper to extract feats + optionally classifier predictions
    def _get_reid_predictions(self, frame, bboxes):
        """
        returns (feats, pred_labels, pred_confs)
        - feats: (N,D) np array or zeros
        - pred_labels: list of int (or -1)
        - pred_confs: list of float (0..1)
        Tries several encoder methods for compatibility:
        1) encoder(frame, list_bboxes) -> feats
        2) encoder.predict_labels(frame, list_bboxes) -> labels, confs (optional)
        3) fallback: color-histogram features
        """
        N = len(bboxes)
        if N == 0:
            return np.zeros((0,256), dtype=np.float32), [-1]*0, [0.0]*0

        # prepare padded pixel crops for fallback or for encoder
        H,W = frame.shape[:2]
        rects = []
        for b in bboxes:
            x1,y1,x2,y2 = int(max(0,b[0])), int(max(0,b[1])), int(min(W-1,b[2])), int(min(H-1,b[3]))
            if x2<=x1: x2 = min(W-1,x1+1)
            if y2<=y1: y2 = min(H-1,y1+1)
            rects.append([x1,y1,x2,y2])

        feats = None
        try:
            # primary: encoder(frame, rects) -> feats
            if self.encoder is not None:
                feats = self.encoder(frame, rects)
                feats = np.asarray(feats, dtype=np.float32)
            else:
                feats = None
        except Exception:
            feats = None

        if feats is None or feats.size == 0:
            # fallback to color-hist features
            tmp = []
            for bb in rects:
                x1,y1,x2,y2 = bb
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
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

        # normalize
        feats = l2_norm(feats)

        # now try to obtain pred labels/confidences if encoder supports it
        pred_labels = [-1]*N
        pred_confs = [0.0]*N
        try:
            if self.encoder is not None and hasattr(self.encoder, "predict_labels"):
                labs, confs = self.encoder.predict_labels(frame, rects)
                # labs might be torch tensor
                labs = np.array(labs).astype(int).tolist()
                confs = np.array(confs).astype(float).tolist()
                # if returned single label+conf per crop, use them
                if len(labs) == N:
                    pred_labels = labs
                if len(confs) == N:
                    pred_confs = confs
        except Exception:
            pass

        return feats, pred_labels, pred_confs

    def predict(self):
        # increment ages
        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

    def _compute_cost(self, active_tracks: List[Track], dets: List[List[float]], det_feats: np.ndarray):
        M = len(active_tracks); N = len(dets)
        if M == 0 or N == 0:
            return np.empty((M,N), dtype=float)
        track_embs = np.vstack([tr.get_tracklet_embedding() for tr in active_tracks])
        app_cost = cosine_distance_matrix(track_embs, det_feats) if track_embs.size and det_feats.size else np.ones((M,N), dtype=float)
        motion_cost = np.zeros((M,N), dtype=float)
        iou_cost = np.zeros((M,N), dtype=float)
        for i,tr in enumerate(active_tracks):
            for j,db in enumerate(dets):
                d = center_distance(tr.bbox, db)
                motion_cost[i,j] = min(1.0, float(d) / (self.motion_thresh + EPS))
                iou_cost[i,j] = 1.0 - bbox_iou(tr.bbox, db)
        cost = self.w_app * app_cost + self.w_motion * motion_cost + self.w_iou * iou_cost
        return cost

    def update(self, results, frame):
        """
        results: Nx6 array (x1,y1,x2,y2,score,cls)
        frame: BGR image
        returns: list of outputs per tracked object: [x1,y1,x2,y2, display_id, score, fish_label]
        """
        self.frame_id += 1
        # parse detections
        if isinstance(results, np.ndarray):
            if results.size == 0:
                dets = []; scores = []
            else:
                dets = [list(r[:4]) for r in results]
                scores = [float(r[4]) for r in results]
        else:
            dets = [list(r[:4]) for r in results]
            scores = [float(r[4]) for r in results]

        if len(dets) == 0:
            self.predict()
            outs=[]
            for tr in self.tracks:
                if tr.state == 'Tracked':
                    display = (tr.fish_label + 1) if tr.fish_label >= 0 else tr.track_id
                    outs.append([float(tr.bbox[0]), float(tr.bbox[1]), float(tr.bbox[2]), float(tr.bbox[3]), int(display), float(tr.score), int(tr.fish_label)])
            return np.asarray(outs, dtype=object)

        # get reid predictions
        det_feats, pred_labels, pred_confs = self._get_reid_predictions(frame, dets)
        if det_feats.ndim == 1:
            det_feats = det_feats.reshape(1,-1)

        # prediction/tick
        self.predict()

        # label-priority matching: if a detection has pred_conf >= C_cls, try to match to track with same fish_label
        matched_det = set()
        for di,(pl,pc) in enumerate(zip(pred_labels, pred_confs)):
            if pl < 0 or pc < self.C_cls:
                continue
            # find tracked track with this fish_label
            mapped = None
            for tr in self.tracks:
                if tr.fish_label == pl and tr.state == 'Tracked':
                    mapped = tr; break
            if mapped is not None:
                if center_distance(mapped.bbox, dets[di]) <= (self.motion_thresh * 1.5):
                    mapped.update_on_match(dets[di], det_feats[di] if det_feats.size else None, scores[di], self.frame_id)
                    mapped.add_label_vote(pl, pc)
                    matched_det.add(di)
                    continue
            # try lost tracks with same fish_label
            for lost_tr in list(self.lost_tracks):
                if lost_tr.fish_label == pl:
                    if (self.frame_id - lost_tr.last_frame) <= (self.reserve_frames + self.max_age):
                        if center_distance(lost_tr.bbox, dets[di]) <= (self.motion_thresh * 1.8):
                            lost_tr.update_on_match(dets[di], det_feats[di] if det_feats.size else None, scores[di], self.frame_id)
                            try:
                                self.lost_tracks.remove(lost_tr)
                            except Exception:
                                pass
                            self.tracks.append(lost_tr)
                            lost_tr.add_label_vote(pl, pc)
                            matched_det.add(di)
                            break

        active_tracks = [tr for tr in self.tracks if tr.state == 'Tracked']
        unmatched_idx = [i for i in range(len(dets)) if i not in matched_det]

        # cost matching for active_tracks vs unmatched detections
        cost = self._compute_cost(active_tracks, [dets[i] for i in unmatched_idx], det_feats[unmatched_idx]) if len(active_tracks)>0 and len(unmatched_idx)>0 else np.empty((len(active_tracks), len(unmatched_idx)))
        matches = []
        if cost.size != 0:
            cost_mat = cost.copy()
            matched_t=set(); matched_d=set()
            while True:
                if cost_mat.size == 0:
                    break
                idx = np.unravel_index(np.argmin(cost_mat), cost_mat.shape)
                i,j = int(idx[0]), int(idx[1])
                if cost_mat[i,j] >= 1e8:
                    break
                det_global_idx = unmatched_idx[j]
                matches.append((i, det_global_idx))
                matched_t.add(i); matched_d.add(j)
                cost_mat[i,:] = 1e9
                cost_mat[:,j] = 1e9
            unmatched_tracks_idx = [i for i in range(len(active_tracks)) if i not in matched_t]
            unmatched_dets_idx = [unmatched_idx[j] for j in range(len(unmatched_idx)) if j not in matched_d]
        else:
            unmatched_tracks_idx = list(range(len(active_tracks)))
            unmatched_dets_idx = unmatched_idx.copy()

        # apply matches: update tracks
        for ai, det_idx in matches:
            tr = active_tracks[ai]
            tr.update_on_match(dets[det_idx], det_feats[det_idx] if det_feats.size else None, scores[det_idx], self.frame_id)
            pl = pred_labels[det_idx] if det_idx < len(pred_labels) else -1
            pc = pred_confs[det_idx] if det_idx < len(pred_confs) else 0.0
            if pl >= 0:
                tr.add_label_vote(pl, pc)

        # reactivation from lost tracks by appearance
        if len(self.lost_tracks)>0 and len(unmatched_dets_idx)>0 and det_feats.size != 0:
            lost_list = list(self.lost_tracks)
            lost_feats = np.vstack([tr.get_tracklet_embedding() for tr in lost_list]) if len(lost_list)>0 else np.zeros((0, det_feats.shape[1] if det_feats.size else 256))
            sub_feats = det_feats[unmatched_dets_idx]
            if lost_feats.size!=0 and sub_feats.size!=0:
                cost_lost = cosine_distance_matrix(lost_feats, sub_feats)
                used_lost=set(); used_det=set()
                while True:
                    if cost_lost.size==0:
                        break
                    li,cj = np.unravel_index(np.argmin(cost_lost), cost_lost.shape)
                    val = float(cost_lost[li,cj])
                    if val >= 1e8:
                        break
                    det_global_idx = unmatched_dets_idx[cj]
                    # allow reactivation if similar enough AND motion reasonable
                    if val <= (1.0 - self.C_emb) and center_distance(lost_list[li].bbox, dets[det_global_idx]) <= (self.motion_thresh * 1.5):
                        lost_tr = lost_list[li]
                        lost_tr.update_on_match(dets[det_global_idx], det_feats[det_global_idx] if det_feats.size else None, scores[det_global_idx], self.frame_id)
                        try:
                            if lost_tr in self.lost_tracks:
                                self.lost_tracks.remove(lost_tr)
                        except Exception:
                            pass
                        self.tracks.append(lost_tr)
                        used_lost.add(li); used_det.add(cj)
                    cost_lost[li,:] = 1e9
                    cost_lost[:,cj] = 1e9
                reactivated = set([unmatched_dets_idx[c] for c in used_det])
                unmatched_dets_idx = [d for d in unmatched_dets_idx if d not in reactivated]

        # fallback center-distance matching
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
                    unmatched_dets_idx.remove(best_j)

        # remaining unmatched detections -> create temporary tracks if allowed
        currently_matched = set([m[1] for m in matches]) | matched_det
        remaining = [i for i in range(len(dets)) if i not in currently_matched]
        new_tracks_created = []
        if len(remaining) > 0:
            if self.allow_new_tracks:
                for d_idx in remaining:
                    tr = Track(dets[d_idx], det_feats[d_idx] if det_feats.size else None, scores[d_idx], self._next_id, self.frame_id, is_temporary=True, tracklet_len=self.tracklet_len)
                    self._next_id += 1
                    self.tracks.append(tr)
                    new_tracks_created.append(tr)

        # convert stable temporary tracks to permanent fish_label if votes or embedding stability indicate
        for tr in list(self.tracks):
            if tr.is_temporary:
                # either label votes strong OR repeated classifier pred on tracklet
                lbl_counts = Counter(tr.label_votes)
                if len(lbl_counts) > 0:
                    lbl, cnt = lbl_counts.most_common(1)[0]
                else:
                    lbl, cnt = -1, 0
                if tr.hits >= self.stable_frames and cnt >= max(2, int(self.stable_frames/2)):
                    # adopt label as fish_label
                    if lbl >= 0:
                        tr.fish_label = int(lbl)
                        tr.is_temporary = False

        # move tracks not updated this frame to lost
        to_move = []
        for tr in list(self.tracks):
            if tr.time_since_update > 0:
                tr.state = 'Lost'
                to_move.append(tr)
        for tr in to_move:
            try:
                self.tracks.remove(tr)
                self.lost_tracks.append(tr)
            except Exception:
                pass

        # cleanup very old lost tracks
        to_delete = [tr for tr in self.lost_tracks if (self.frame_id - tr.last_frame) > (self.reserve_frames + self.max_age)]
        for tr in to_delete:
            try:
                self.lost_tracks.remove(tr)
                tr.state = 'Removed'
                self.removed_tracks.append(tr)
            except Exception:
                pass

        # diagnostics writing and output assembly
        outs=[]
        if self.diagnostics_csv:
            try:
                with open(self.diagnostics_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    for di in range(len(dets)):
                        x1,y1,x2,y2 = dets[di]
                        score = scores[di] if di < len(scores) else 0.0
                        pl = pred_labels[di] if di < len(pred_labels) else -1
                        pc = pred_confs[di] if di < len(pred_confs) else 0.0
                        t_ids=[]; t_dists=[]
                        if len(self.tracks)>0 and det_feats.size!=0:
                            act_feats = np.vstack([tr.get_tracklet_embedding() for tr in self.tracks])
                            idxs = np.argsort(cosine_distance_matrix(act_feats, det_feats[di:di+1]).flatten())[:3]
                            for ii in idxs:
                                if ii < len(self.tracks):
                                    t_ids.append(self.tracks[ii].track_id)
                                    t_dists.append(float(cosine_distance_matrix(act_feats[ii:ii+1], det_feats[di:di+1]).flatten()[0]))
                        cdist=-1.0; ioubest=0.0
                        if len(self.tracks)>0:
                            dlist = [center_distance(tr.bbox, dets[di]) for tr in self.tracks]
                            cdist = float(np.min(dlist))
                            ioubest = float(np.max([bbox_iou(tr.bbox, dets[di]) for tr in self.tracks]))
                        # matched track if any whose bbox roughly equals det bbox
                        matched_tid = None; matched_display=None
                        for tr in self.tracks:
                            if all(abs(tr.bbox[i] - dets[di][i]) < 2.0 for i in range(4)):
                                matched_tid = tr.track_id
                                matched_display = (tr.fish_label + 1) if tr.fish_label >= 0 else tr.track_id
                                break
                        writer.writerow([self.frame_id, di, x1,y1,x2,y2, score, pl, pc, t_ids, t_dists, cdist, ioubest, matched_tid, matched_display])
            except Exception:
                pass

        for tr in self.tracks:
            if tr.state == 'Tracked':
                display = (tr.fish_label + 1) if tr.fish_label >= 0 else tr.track_id
                outs.append([float(tr.bbox[0]), float(tr.bbox[1]), float(tr.bbox[2]), float(tr.bbox[3]), int(display), float(tr.score), int(tr.fish_label)])
        return np.asarray(outs, dtype=object)
