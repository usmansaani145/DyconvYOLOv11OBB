#!/usr/bin/env python3
"""Run MMRotate tools/test.py with rotated NMS CPU/GPU mismatch patch."""

from __future__ import annotations

import importlib
import runpy
import sys


def install_nms_patch() -> None:
    import torch
    from mmcv.ops import nms_rotated

    bn = importlib.import_module("mmrotate.core.post_processing.bbox_nms_rotated")
    orig = bn.multiclass_nms_rotated

    def multiclass_nms_rotated_fixed(
        multi_bboxes,
        multi_scores,
        score_thr,
        nms,
        max_num=-1,
        score_factors=None,
        return_inds=False,
    ):
        if not isinstance(multi_bboxes, torch.Tensor) or not isinstance(
            multi_scores, torch.Tensor
        ):
            return orig(
                multi_bboxes,
                multi_scores,
                score_thr,
                nms,
                max_num=max_num,
                score_factors=score_factors,
                return_inds=return_inds,
            )

        device = multi_scores.device
        multi_bboxes = multi_bboxes.to(device)
        if score_factors is not None and isinstance(score_factors, torch.Tensor):
            score_factors = score_factors.to(device)

        num_classes = multi_scores.size(1) - 1
        if multi_bboxes.size(1) > 5:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, :num_classes, :]
        else:
            bboxes = multi_bboxes[:, None, :].expand(multi_scores.size(0), num_classes, 5)

        scores = multi_scores[:, :-1]
        if score_factors is not None:
            scores = scores * score_factors[:, None]

        bboxes = bboxes.reshape(-1, 5)
        scores = scores.reshape(-1)
        labels = (
            torch.arange(num_classes, device=device)
            .view(1, -1)
            .expand(multi_scores.size(0), num_classes)
            .reshape(-1)
        )

        valid = scores > float(score_thr)
        if valid.sum() == 0:
            empty_bboxes = bboxes.new_zeros((0, 6))
            empty_labels = labels.new_zeros((0,), dtype=torch.long)
            if return_inds:
                empty_inds = labels.new_zeros((0,), dtype=torch.long)
                return empty_bboxes, empty_labels, empty_inds
            return empty_bboxes, empty_labels

        bboxes = bboxes[valid]
        scores = scores[valid]
        labels = labels[valid]

        if isinstance(nms, dict):
            iou_thr = float(nms.get("iou_thr", 0.1))
        else:
            iou_thr = float(getattr(nms, "iou_thr", 0.1))

        keep_mask = []
        for c in range(num_classes):
            cls_inds = torch.nonzero(labels == c, as_tuple=False).squeeze(1)
            if cls_inds.numel() == 0:
                continue
            cls_b = bboxes[cls_inds]
            cls_s = scores[cls_inds]
            _, cls_keep = nms_rotated(cls_b, cls_s, iou_thr)
            keep_mask.append(cls_inds[cls_keep])

        if len(keep_mask) == 0:
            empty_bboxes = bboxes.new_zeros((0, 6))
            empty_labels = labels.new_zeros((0,), dtype=torch.long)
            if return_inds:
                empty_inds = labels.new_zeros((0,), dtype=torch.long)
                return empty_bboxes, empty_labels, empty_inds
            return empty_bboxes, empty_labels

        keep = torch.cat(keep_mask, dim=0)
        det_bboxes = torch.cat([bboxes[keep], scores[keep, None]], dim=1)
        det_labels = labels[keep]

        order = det_bboxes[:, -1].sort(descending=True).indices
        det_bboxes = det_bboxes[order]
        det_labels = det_labels[order]

        if max_num is not None and int(max_num) > 0:
            det_bboxes = det_bboxes[: int(max_num)]
            det_labels = det_labels[: int(max_num)]

        return det_bboxes, det_labels

    bn.multiclass_nms_rotated = multiclass_nms_rotated_fixed
    importlib.import_module("mmrotate.core.post_processing").multiclass_nms_rotated = (
        multiclass_nms_rotated_fixed
    )
    importlib.import_module("mmrotate.core").multiclass_nms_rotated = multiclass_nms_rotated_fixed

    for mod_name in (
        "mmrotate.models.roi_heads.bbox_heads.rotated_bbox_head",
        "mmrotate.models.dense_heads.rotated_anchor_head",
    ):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "multiclass_nms_rotated"):
                mod.multiclass_nms_rotated = multiclass_nms_rotated_fixed
        except Exception:
            pass


def main() -> None:
    install_nms_patch()
    sys.argv[0] = "/opt/mmrotate/tools/test.py"
    runpy.run_path("/opt/mmrotate/tools/test.py", run_name="__main__")


if __name__ == "__main__":
    main()
