# Ultralytics YOLO 🚀, AGPL-3.0 license

import importlib.util
import math
from collections import Counter, deque
from typing import List, Optional

import numpy as np

from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """
    Extended STrack with ReID voting (S1), permanent ID (S4), and display-id management (S5).
    Returns 10-column results: [x1,y1,x2,y2,display_id,score,cls,idx,perm_id,track_id]
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50, reid_conf=None):
        super().__init__(tlwh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        # S1 voting fields
        self.reid_conf = reid_conf
        self.cls_history = deque()
        self.label_votes = deque(maxlen=15)
        self.fish_label = -1
        self.hits = 0

        # S4 fields
        self.perm_id = None
        self.id_locked = False
        self.is_temporary = True

        # Speed record (not used for cost)
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
        """Predict state with Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter, frame_id):
        if self.cls is not None:
            self.cls_history.append(int(self.cls))
        super().activate(kalman_filter, frame_id)
        self.hits = 1
        self._update_speed()
        if self.cls is not None:
            self.add_label_vote(self.cls)

    def re_activate(self, new_track, frame_id, new_id=False):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        if new_track.cls is not None:
            self.cls_history.append(int(new_track.cls))
        super().re_activate(new_track, frame_id, new_id)
        self.hits += 1
        self._update_speed()
        if self.cls is not None:
            self.add_label_vote(self.cls)

    def update(self, new_track, frame_id):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        if new_track.cls is not None:
            self.cls_history.append(int(new_track.cls))
        super().update(new_track, frame_id)
        self.hits += 1
        self._update_speed()
        if self.cls is not None:
            self.add_label_vote(self.cls)

    def add_label_vote(self, label: int):
        """Accumulate ReID/class votes and update fish_label by majority."""
        if label is None:
            return
        try:
            label_int = int(label)
        except Exception:
            return
        if label_int < 0:
            return
        self.label_votes.append(label_int)
        ctr = Counter(self.label_votes)
        if len(ctr) > 0:
            lbl, _ = ctr.most_common(1)[0]
            self.fish_label = int(lbl)

    def promote_to_permanent(self, perm_id: int):
        """S4: promote to permanent representative (1..MAX_FISH)."""
        self.is_temporary = False
        self.perm_id = int(perm_id)
        self.id_locked = True

    def _update_speed(self):
        tlwh = self.tlwh
        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]
            self.speed = math.hypot(dx, dy)
        self.prev_center = (cx, cy)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
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
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def result(self):
        coords = self.xyxy if self.angle is None else self.xywha

        display_id = getattr(self, "display_id", None)
        if display_id is None:
            display_id = int(self.track_id)
            if self.perm_id is not None:
                display_id = int(self.perm_id)
            elif self.fish_label is not None and self.fish_label >= 0:
                display_id = int(self.fish_label) + 1
        else:
            display_id = int(display_id)

        perm = -1 if self.perm_id is None else int(self.perm_id)

        return coords.tolist() + [
            display_id,
            self.score,
            self.cls,
            self.idx,
            perm,
            int(self.track_id),
        ]


class BOTSORT(BYTETracker):
    """
    BoT-SORT variant with:
    - S1: fish_label voting via ReID classifier
    - S4: permanent ID promotion/takeover based on stable votes
    - S5: per-frame unique display_id assignment
    """

    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        # thresholds
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        # S1/S4 params
        self.MAX_FISH = 9
        self.stable_frames = getattr(args, "stable_frames", 5)
        self.min_votes = getattr(args, "min_votes", 3)
        self.takeover_hits_margin = getattr(args, "takeover_hits_margin", 0)

        # S2 params
        self.cls_mismatch_penalty = float(getattr(args, "cls_mismatch_penalty", 0.07))
        self.cls_conf_thresh = float(getattr(args, "cls_conf_thresh", 0.6))

        # perm_id tracking
        self.used_perm_ids = set()
        self.permanent_tracks = {}

        # ReID encoder
        self.encoder = None
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
        self.gmc = GMC(method=args.gmc_method)

    def update(self, results, img=None):
        if img is not None:
            self.img_h, self.img_w = img.shape[:2]

        super().update(results, img)

        self._update_permanent_tracks()
        self._assign_display_ids()

        outputs = []
        for track in self.tracked_stracks:
            if track.is_activated:
                outputs.append(track.result)

        if len(outputs) == 0:
            return np.zeros((0, 10), dtype=float)
        return np.asarray(outputs, dtype=float)

    def get_kalmanfilter(self):
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        if len(dets) == 0:
            return []

        features_keep, reid_labels, reid_confs = None, None, None
        if self.encoder is not None and img is not None:
            xyxy = xywh2xyxy(np.asarray(dets)[:, :4])
            try:
                features_keep, reid_labels, reid_confs = self.encoder.encode_and_classify(img, xyxy)
            except AttributeError:
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

    def _stable_label(self, track: BOTrack):
        fish_label = getattr(track, "fish_label", None)
        if fish_label is None or fish_label < 0 or fish_label >= self.MAX_FISH:
            return None, 0
        votes = getattr(track, "label_votes", None)
        try:
            support = Counter(votes).get(fish_label, 0) if votes is not None else 0
        except Exception:
            try:
                support = list(votes).count(fish_label)
            except Exception:
                support = 0
        if track.hits >= self.stable_frames and support >= self.min_votes:
            return fish_label, support
        return None, support

    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        if getattr(self.args, "fuse_score", False):
            dists = matching.fuse_score(dists, detections)

        use_reid = getattr(self.args, "with_reid", False) and self.encoder is not None
        if use_reid and len(tracks) > 0 and len(detections) > 0:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)

            if dists.size > 0:
                for ti, tr in enumerate(tracks):
                    fish_label, _ = self._stable_label(tr)
                    if fish_label is None:
                        continue
                    for dj, det in enumerate(detections):
                        if dj >= dists.shape[1]:
                            break
                        try:
                            det_cls = int(getattr(det, "cls", -1))
                        except Exception:
                            det_cls = -1
                        if det_cls < 0:
                            continue
                        det_conf = getattr(det, "reid_conf", None)
                        if det_conf is not None and det_conf < self.cls_conf_thresh:
                            continue
                        if det_cls != fish_label:
                            dists[ti, dj] = min(1.0, float(dists[ti, dj]) + self.cls_mismatch_penalty)
        return dists

    def multi_predict(self, tracks):
        BOTrack.multi_predict(tracks)

    def reset(self):
        super().reset()
        self.gmc.reset_params()
        self.used_perm_ids = set()
        self.permanent_tracks = {}

    def _update_permanent_tracks(self):
        """S4: promote stable tracks to perm_id (1..MAX_FISH), allow takeover if stronger/newer."""
        for track in self.tracked_stracks:
            if not hasattr(track, "fish_label"):
                continue
            if track.fish_label is None or track.fish_label < 0:
                continue

            support = 0
            if hasattr(track, "label_votes"):
                try:
                    support = Counter(track.label_votes).get(track.fish_label, 0)
                except Exception:
                    try:
                        support = list(track.label_votes).count(track.fish_label)
                    except Exception:
                        support = 0

            if track.hits < self.stable_frames or support < self.min_votes:
                continue

            candidate_perm_id = int(track.fish_label) + 1
            if candidate_perm_id < 1 or candidate_perm_id > self.MAX_FISH:
                continue

            existing = self.permanent_tracks.get(candidate_perm_id)

            if not track.is_temporary:
                if existing is None or existing is track:
                    self.permanent_tracks[candidate_perm_id] = track
                    self.used_perm_ids.add(candidate_perm_id)
                continue

            if existing is None:
                track.promote_to_permanent(candidate_perm_id)
                self.permanent_tracks[candidate_perm_id] = track
                self.used_perm_ids.add(candidate_perm_id)
                continue

            too_old = (self.frame_id - existing.end_frame) > self.max_time_lost if hasattr(self, "max_time_lost") else False
            margin = getattr(self, "takeover_hits_margin", 0)
            old_hits = getattr(existing, "hits", 0)
            new_hits = getattr(track, "hits", 0)
            better_hits = new_hits >= (old_hits + margin)

            if existing.state != TrackState.Tracked or too_old or better_hits:
                existing.is_temporary = True
                existing.perm_id = None

                track.promote_to_permanent(candidate_perm_id)
                self.permanent_tracks[candidate_perm_id] = track
                self.used_perm_ids.add(candidate_perm_id)

    def _assign_display_ids(self):
        """S5: per-frame unique display_id assignment (perm_id priority, then stable fish_label, then fallback)."""
        if not self.tracked_stracks:
            return

        for t in self.tracked_stracks:
            if hasattr(t, "display_id"):
                t.display_id = None

        used_ids = set()
        candidates_by_fish = {k: [] for k in range(self.MAX_FISH)}

        # perm_id priority
        for t in self.tracked_stracks:
            perm = getattr(t, "perm_id", None)
            if perm is None or perm <= 0:
                continue
            did = int(perm)
            if did in used_ids:
                continue
            t.display_id = did
            used_ids.add(did)

        # one representative per fish_label (stable)
        for t in self.tracked_stracks:
            if getattr(t, "display_id", None) is not None:
                continue
            fish_label = getattr(t, "fish_label", None)
            if fish_label is None or fish_label < 0:
                continue
            k = int(fish_label)
            if not (0 <= k < self.MAX_FISH):
                continue

            votes = getattr(t, "label_votes", None)
            try:
                support = Counter(votes).get(fish_label, 0) if votes is not None else 0
            except Exception:
                try:
                    support = list(votes).count(fish_label)
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
            if fish_id in used_ids:
                continue
            if not cand_list:
                continue
            cand_list.sort(key=lambda x: x[0], reverse=True)
            best_track = cand_list[0][1]
            best_track.display_id = fish_id
            used_ids.add(fish_id)

        # fallback to unique track-based IDs avoiding conflicts
        for t in self.tracked_stracks:
            if getattr(t, "display_id", None) is not None:
                continue
            base = int(t.track_id) + self.MAX_FISH
            did = base
            while did in used_ids:
                did += self.MAX_FISH
            t.display_id = did
            used_ids.add(did)
