"""SAHI-style sliding-window tile origins for Figure 2."""

from __future__ import annotations


def get_tile_bboxes(
    img_h: int,
    img_w: int,
    tile_h: int,
    tile_w: int,
    overlap_h: float,
    overlap_w: float,
) -> list[tuple[int, int]]:
    """Return top-left (x, y) for each tile window."""
    stride_h = max(1, int(tile_h * (1.0 - overlap_h)))
    stride_w = max(1, int(tile_w * (1.0 - overlap_w)))
    bboxes: list[tuple[int, int]] = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            bboxes.append((x, y))
            if x + tile_w >= img_w:
                break
            x += stride_w
        if y + tile_h >= img_h:
            break
        y += stride_h
    return bboxes
