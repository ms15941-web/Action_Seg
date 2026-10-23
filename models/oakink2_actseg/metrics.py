"""MS-TCN action segmentation metrics (same definitions as DiffAct/utils.py func_eval):
frame accuracy over all frames, segmental Edit and F1@{10,25,50} ignoring background."""
from collections import OrderedDict

import numpy as np

from .common import BACKGROUND

OVERLAPS = (0.1, 0.25, 0.5)


def _segments(labels, bg):
    # Replicates MS-TCN's get_labels_start_end_time, including the last segment ending at T-1.
    names, starts, ends = [], [], []
    last = labels[0]
    if last not in bg:
        names.append(last)
        starts.append(0)
    for i, lab in enumerate(labels):
        if lab != last:
            if lab not in bg:
                names.append(lab)
                starts.append(i)
            if last not in bg:
                ends.append(i)
            last = lab
    if last not in bg:
        ends.append(len(labels) - 1)
    return names, np.array(starts), np.array(ends)


def _edit(pred, gt, bg):
    p, _, _ = _segments(pred, bg)
    y, _, _ = _segments(gt, bg)
    D = np.zeros((len(p) + 1, len(y) + 1))
    D[:, 0] = np.arange(len(p) + 1)
    D[0, :] = np.arange(len(y) + 1)
    for j in range(1, len(y) + 1):
        for i in range(1, len(p) + 1):
            D[i, j] = D[i - 1, j - 1] if y[j - 1] == p[i - 1] else 1 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return (1 - D[-1, -1] / max(len(p), len(y), 1)) * 100


def _f_counts(pred, gt, overlap, bg):
    p_lab, p_s, p_e = _segments(pred, bg)
    y_lab, y_s, y_e = _segments(gt, bg)
    tp = fp = 0
    hits = np.zeros(len(y_lab))
    for j in range(len(p_lab)):
        if len(y_lab) == 0:
            fp += 1
            continue
        inter = np.minimum(p_e[j], y_e) - np.maximum(p_s[j], y_s)
        union = np.maximum(p_e[j], y_e) - np.minimum(p_s[j], y_s)
        iou = inter / union * np.array([p_lab[j] == y for y in y_lab])
        idx = iou.argmax()
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    return tp, fp, len(y_lab) - hits.sum()


def evaluate(gt, pred, videos, bg=(BACKGROUND,)):
    """gt / pred: {video: [label per frame]} with string labels."""
    correct = total = 0
    edit = 0.0
    counts = np.zeros((len(OVERLAPS), 3))
    for v in videos:
        g, p = gt[v], pred[v]
        if len(g) != len(p):
            raise ValueError(f"{v}: {len(p)} predicted frames vs {len(g)} ground-truth frames")
        correct += sum(a == b for a, b in zip(g, p))
        total += len(g)
        edit += _edit(p, g, bg)
        for k, ov in enumerate(OVERLAPS):
            counts[k] += _f_counts(p, g, ov, bg)
    res = OrderedDict(Acc=100.0 * correct / total, Edit=edit / len(videos))
    for k, ov in enumerate(OVERLAPS):
        tp, fp, fn = counts[k]
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        res[f"F1@{int(ov * 100)}"] = 100.0 * f1
    return res
