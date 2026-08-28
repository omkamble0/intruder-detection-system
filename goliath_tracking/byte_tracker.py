"""
GOLIATH Multi-Object Tracking - ByteTrack Engine
Implements two-stage data association with 8D Kalman filter state-space estimation.
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum

from goliath_tracking.kalman_filter import KalmanFilter
from goliath_tracking.matching import compute_ious, linear_assignment

class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """
    Single Track representation holding Kalman state distribution,
    spatial box history, and identity lifecycle.
    """
    _count = 0

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int = 0):
        # [x, y, w, h]
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated = False

        self.score = score
        self.class_id = class_id
        self.tracklet_len = 0
        self.state = TrackState.New

        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> np.ndarray:
        return self.tlwh_to_xyah(self.tlwh)

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Initializes state with Kalman filter and assigns unique ID."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """Re-activates lost track with newly matched measurement."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track: 'STrack', frame_id: int):
        """Updates tracked instance with new measurement."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class ByteTracker:
    """
    Enterprise ByteTrack implementation with two-stage association.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        low_conf_thresh: float = 0.1,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_conf_thresh = low_conf_thresh
        self.track_buffer = track_buffer
        self.kalman_filter = KalmanFilter()

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0

    def reset(self):
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_counter()

    def update(self, output_results: np.ndarray) -> List[STrack]:
        """
        Updates tracks from detector outputs.
        output_results: array of shape (N, 6) where each row is [x1, y1, x2, y2, score, class_id]
        Returns: list of active STrack instances.
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        if len(output_results) > 0:
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]

            # Step 1: Split into High-score and Low-score detections
            remain_inds = scores >= self.track_thresh
            inds_low = (scores >= self.low_conf_thresh) & (scores < self.track_thresh)

            dets = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, int(c))
                for tlbr, s, c in zip(bboxes[remain_inds], scores[remain_inds], classes[remain_inds])
            ]
            dets_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, int(c))
                for tlbr, s, c in zip(bboxes[inds_low], scores[inds_low], classes[inds_low])
            ]
        else:
            dets = []
            dets_second = []

        # Predict current locations with Kalman Filter
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: Kalman prediction
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for strack in strack_pool:
            strack.predict()

        # Step 3: First association with high-score detections
        dists = 1.0 - compute_ious(
            np.array([st.tlbr for st in strack_pool]),
            np.array([st.tlbr for st in dets]) if dets else np.empty((0, 4))
        )
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 4: Second association with low-score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = 1.0 - compute_ious(
            np.array([st.tlbr for st in r_tracked_stracks]),
            np.array([st.tlbr for st in dets_second]) if dets_second else np.empty((0, 4))
        )
        matches, u_r_track, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = dets_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_r_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 5: Initialize new tracks from unmatched high-confidence detections
        for inew in u_detection:
            track = dets[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 6: Update lost / removed tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Return active tracks
        return [track for track in self.tracked_stracks if track.is_activated]


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if t.track_id not in exists:
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())
