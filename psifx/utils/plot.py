from typing import Optional, Tuple

from colorsys import hls_to_rgb

import numpy as np
from skimage import draw


def get_palette(
    n: int,
    hue: float = 0.01,
    luminance: float = 0.6,
    saturation: float = 0.65,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Returns a palette of colours between [0, 1].

    :param n:
    :param hue:
    :param luminance:
    :param saturation:
    :param dtype:
    :return: [N, 3]
    """
    hues = np.linspace(0, 1, n + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(dtype=np.int64)
    palette = [hls_to_rgb(float(hue), luminance, saturation) for hue in hues]
    palette = np.array(palette, dtype=np.float32)
    if dtype == np.uint8:
        palette = (palette * 255.0).astype(dtype=dtype)
    elif dtype == np.uint16:
        palette = (palette * 65535.0).astype(dtype=dtype)
    return palette


def _draw_circles(
    image: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """

    :param image: [H, W, C]
    :param points: [N, 2]
    :param colors: [N, C]
    :param radius:
    :return: [H, W, C]
    """
    assert points.shape[-1] == 2
    assert points.shape[-2] == colors.shape[-2]
    assert image.shape[-1] == colors.shape[-1]

    n, _ = points.shape

    cols, rows = points[:, 0], points[:, 1]

    cols = cols.round().astype(dtype=np.int64)
    rows = rows.round().astype(dtype=np.int64)

    image = image.copy()
    for i in range(n):
        rr, cc, val = draw.circle_perimeter_aa(
            r=rows[i],
            c=cols[i],
            radius=radius,
        )
        draw.set_color(
            image=image,
            coords=(rr, cc),
            color=colors[i],
            alpha=val,
        )

    return image


def draw_circles(
    image: np.ndarray,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: int = 1,
) -> np.ndarray:
    """

    :param image: [*, H, W, C]
    :param points: [*, N, 2]
    :param colors: [N, C]
    :param radius:
    :return: [*, H, W, C]
    """

    assert image.shape[:-3] == points.shape[:-2]
    assert points.shape[-1] == 2

    shape = image.shape

    image = image.reshape((-1,) + image.shape[-3:])
    points = points.reshape((-1,) + points.shape[-2:])

    b, _, _, c = image.shape
    _, n, _ = points.shape

    if colors is None:
        colors = get_palette(n, dtype=image.dtype)

    assert points.shape[-2] == colors.shape[-2]
    assert image.shape[-1] == colors.shape[-1]

    for i in range(b):
        image[i] = _draw_circles(
            image=image[i],
            points=points[i],
            colors=colors,
            radius=radius,
        )

    image = image.reshape(shape)

    return image


def _draw_lines(
    image: np.ndarray,
    start_points: np.ndarray,
    end_points: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """

    :param image: [H, W, C]
    :param start_points: [N, 2]
    :param end_points: [N, 2]
    :param colors: [N, C]
    :return: [H, W, C]
    """
    assert start_points.shape[-1] == end_points.shape[-1] == 2
    assert start_points.shape[-2] == end_points.shape[-2] == colors.shape[-2]
    assert image.shape[-1] == colors.shape[-1]

    n, _ = start_points.shape

    start_cols, start_rows = start_points[:, 0], start_points[:, 1]
    end_cols, end_rows = end_points[:, 0], end_points[:, 1]

    start_cols = start_cols.round().astype(dtype=np.int64)
    start_rows = start_rows.round().astype(dtype=np.int64)
    end_cols = end_cols.round().astype(dtype=np.int64)
    end_rows = end_rows.round().astype(dtype=np.int64)

    image = image.copy()
    for i in range(n):
        rr, cc, val = draw.line_aa(
            r0=start_rows[i],
            c0=start_cols[i],
            r1=end_rows[i],
            c1=end_cols[i],
        )
        draw.set_color(
            image=image,
            coords=(rr, cc),
            color=colors[i],
            alpha=val,
        )

    return image


def draw_lines(
    image: np.ndarray,
    start_points: np.ndarray,
    end_points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """

    :param image: [*, H, W, C]
    :param start_points: [*, N, 2]
    :param end_points: [*, N, 2]
    :param colors: [*, N, C]
    :return: [*, H, W, C]
    """
    assert image.shape[:-3] == start_points.shape[:-2] == end_points.shape[:-2]
    assert start_points.shape[-1] == end_points.shape[-1] == 2

    shape = image.shape

    image = image.reshape((-1,) + image.shape[-3:])
    start_points = start_points.reshape((-1,) + start_points.shape[-2:])
    end_points = end_points.reshape((-1,) + end_points.shape[-2:])

    b, _, _, c = image.shape
    _, n, _ = start_points.shape

    if colors is None:
        colors = get_palette(n, dtype=image.dtype)

    assert start_points.shape[-2] == end_points.shape[-2] == colors.shape[-2]
    assert image.shape[-1] == colors.shape[-1]

    for i in range(b):
        image[i] = _draw_lines(
            image=image[i],
            start_points=start_points[i],
            end_points=end_points[i],
            colors=colors,
        )

    image = image.reshape(shape)

    return image


def draw_pose(
    image: np.ndarray,
    points: np.ndarray,
    confidences: np.ndarray,
    edges: Tuple[Tuple[int, int], ...],
    circle_colors: Optional[np.ndarray] = None,
    line_colors: Optional[np.ndarray] = None,
    radius: int = 1,
    draw_points: bool = True,
    draw_edges: bool = True,
) -> np.ndarray:
    """

    :param image: [*, H, W, C]
    :param points: [*, N, 2]
    :param confidences: [*, N, 1]
    :param edges: [M, 2]
    :param circle_colors: [N, C]
    :param line_colors: [M, C]
    :param radius:
    :param draw_points:
    :param draw_edges:
    :return: [*, H, W, C]
    """
    assert image.shape[:-3] == points.shape[:-2] == confidences.shape[:-2]
    assert points.shape[-2] == confidences.shape[-2]
    assert points.shape[-1] == 2
    assert confidences.shape[-1] == 1

    if draw_edges:
        start, end = list(zip(*edges))

        edge_confidences = confidences[..., start, :] & confidences[..., end, :]
        edge_confidences = np.concatenate([edge_confidences, edge_confidences], axis=-1)

        start_points = points[..., start, :].copy()
        start_points[~edge_confidences] = -1.0

        end_points = points[..., end, :].copy()
        end_points[~edge_confidences] = -1.0

        image = draw_lines(
            image=image,
            start_points=start_points,
            end_points=end_points,
            colors=line_colors,
        )

    if draw_points:
        points = points.copy()
        confidences = np.concatenate([confidences, confidences], axis=-1)
        points[~confidences] = -1.0

        image = draw_circles(
            image=image,
            points=points,
            colors=circle_colors,
            radius=radius,
        )

    return image
