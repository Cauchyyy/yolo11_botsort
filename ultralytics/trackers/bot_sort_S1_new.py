# Ultralytics YOLO 🚀, AGPL-3.0 license

import importlib.util
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
    S1-only track with voting-based fish_label and per-frame display_id assignment (S5).
    Outputs 10 columns: [x1,y1,x2,y2,display_id,score,cls,idx,perm_id(-1),track_id]
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

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
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
        if self.cls is not None:
            self.add_label_vote(self.cls)

    def re_activate(self, new_track, frame_id, new_id=False):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        if new_track.cls is not None:
            self.cls_history.append(int(new_track.cls))
        super().re_activate(new_track, frame_id, new_id)
        self.hits += 1
        if self.cls is not None:
            self.add_label_vote(self.cls)

    def update(self, new_track, frame_id):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        if new_track.cls is not None:
            self.cls_history.append(int(new_track.cls))
        super().update(new_track, frame_id)
        self.hits += 1
        if self.cls is not None:
            self.add_label_vote(self.cls)

    def add_label_vote(self, label: int):
        if label is None:
            return
        try:
            lbl = int(label)
        except Exception:
            return
        if lbl < 0:
            return
        self.label_votes.append(lbl)
        ctr = Counter(self.label_votes)
        if len(ctr) > 0:
            self.fish_label = int(ctr.most_common(1)[0][0])

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
            if self.fish_label is not None and self.fish_label >= 0:
                display_id = int(self.fish_label) + 1
            else:
                display_id = int(self.track_id)
        else:
            display_id = int(display_id)

        perm = -1  # S4 removed; keep column for compatibility

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
    BoT-SORT with S1 voting and S5 per-frame unique display IDs. No S4 permanent IDs.
    """

    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        # S1 parameters
        self.MAX_FISH = 9
        self.stable_frames = getattr(args, "stable_frames", 5)
        self.min_votes = getattr(args, "min_votes", 3)

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
        return dists

    def multi_predict(self, tracks):
        BOTrack.multi_predict(tracks)

    def reset(self):
        super().reset()
        self.gmc.reset_params()

    def _assign_display_ids(self):
        """S5: per-frame unique display_id assignment."""
        if not self.tracked_stracks:
            return

        for t in self.tracked_stracks:
            if hasattr(t, "display_id"):
                t.display_id = None

        used_ids = set()
        candidates_by_fish = {k: [] for k in range(self.MAX_FISH)}

        # choose stable representative per fish_label
        for t in self.tracked_stracks:
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
            if not cand_list:
                continue
            cand_list.sort(key=lambda x: x[0], reverse=True)
            best_track = cand_list[0][1]
            if fish_id not in used_ids:
                best_track.display_id = fish_id
                used_ids.add(fish_id)

        # fallback: unique track-based IDs avoiding conflicts
        for t in self.tracked_stracks:
            if getattr(t, "display_id", None) is not None:
                continue
            base = int(t.track_id)
            did = base
            while did in used_ids:
                did += self.MAX_FISH
            t.display_id = did
            used_ids.add(did)
