"""
GOLIATH Multi-Object Tracking - Data Association & Cost Matrix Utilities
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

def compute_ious(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Intersection over Union (IoU) between two sets of bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if atlbrs.size == 0 or btlbrs.size == 0:
        return ious

    atlbrs = np.ascontiguousarray(atlbrs, dtype=float)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=float)

    for i, a in enumerate(atlbrs):
        x1 = np.maximum(a[0], btlbrs[:, 0])
        y1 = np.maximum(a[1], btlbrs[:, 1])
        x2 = np.minimum(a[2], btlbrs[:, 2])
        y2 = np.minimum(a[3], btlbrs[:, 3])

        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        intersection = w * h

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
        union = area_a + area_b - intersection

        ious[i] = np.where(union > 0, intersection / union, 0.0)

    return ious


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Solves optimal bipartite matching on cost matrix with threshold gating.
    Returns:
        matches: array of shape (N, 2) where matches[:, 0] = track_idx, matches[:, 1] = det_idx
        unmatched_a: indices of unassigned tracks
        unmatched_b: indices of unassigned detections
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_a, unmatched_b = [], [], []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
        else:
            unmatched_a.append(r)
            unmatched_b.append(c)

    matched_rows = {m[0] for m in matches}
    matched_cols = {m[1] for m in matches}

    for i in range(cost_matrix.shape[0]):
        if i not in matched_rows and i not in unmatched_a:
            unmatched_a.append(i)

    for j in range(cost_matrix.shape[1]):
        if j not in matched_cols and j not in unmatched_b:
            unmatched_b.append(j)

    return np.array(matches, dtype=int).reshape(-1, 2), unmatched_a, unmatched_b
