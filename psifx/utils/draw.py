from typing import Optional, Tuple

from colorsys import hls_to_rgb
import math

import numpy as np
from numpy import ndarray
from skimage import draw


def get_palette(
    n: int,
    hue: float = 0.01,
    luminance: float = 0.6,
    saturation: float = 0.65,
    dtype: np.dtype = np.float32,
) -> ndarray:
    """
    Returns a palette of colours between [0, 1].

    :param n:
    :param hue:
    :param luminance:
    :param saturation:
    :param dtype:
    :return: [N, 3]
    """
    hues = np.linspace(0.0, 1.0, n + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(dtype=np.int64)
    hues = hues.tolist()
    palette = np.array([hls_to_rgb(hue, luminance, saturation) for hue in hues])
    if dtype == np.uint8:
        palette = (palette * 255.0).astype(dtype=dtype)
    return palette


def _draw_circles(
    image: ndarray,
    points: ndarray,
    colors: ndarray,
    radius: int = 1,
) -> ndarray:
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

    h, w, _ = image.shape
    n, _ = points.shape

    image = image.copy()
    cols = points[:, 0]
    rows = points[:, 1]

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


def _draw_disks(
    image: ndarray,
    points: ndarray,
    colors: ndarray,
    radius: int = 1,
) -> ndarray:
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

    h, w, _ = image.shape
    n, _ = points.shape

    image = image.copy()
    cols = points[:, 0]
    rows = points[:, 1]

    for i in range(n):
        rr, cc = draw.disk(
            center=(rows[i], cols[i]),
            radius=radius,
            shape=(h, w),
        )
        image[rr, cc] = colors[i]

    return image


def draw_points(
    image: ndarray,
    points: ndarray,
    colors: Optional[ndarray] = None,
    radius: int = 1,
    mode: str = "circle",
) -> ndarray:
    """

    :param image: [*, H, W, C]
    :param points: [*, N, 2]
    :param colors: [N, C]
    :param radius:
    :param mode: "circle" or "disk"
    :return: [*, H, W, C]
    """
    assert image.shape[:-3] == points.shape[:-2]
    assert points.shape[-1] == 2
    assert mode in ["circle", "disk"]

    if mode == "circle":
        _draw_func = _draw_circles
    else:
        _draw_func = _draw_disks

    shape = image.shape
    dtype = image.dtype

    image = image.reshape((-1,) + image.shape[-3:])
    points = points.reshape((-1,) + points.shape[-2:])
    points = points.round().astype(dtype=np.int64)

    b, _, _, c = image.shape
    _, n, _ = points.shape

    if colors is None:
        colors = get_palette(n, dtype=dtype)

    assert points.shape[-2] == colors.shape[-2]
    assert image.shape[-1] == colors.shape[-1]

    for i in range(b):
        image[i] = _draw_func(
            image=image[i],
            points=points[i],
            colors=colors,
            radius=radius,
        )

    image = image.reshape(shape)

    return image


def _draw_lines(
    image: ndarray,
    start_points: ndarray,
    end_points: ndarray,
    colors: ndarray,
    thickness: int = 1,
) -> ndarray:
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

    h, w, _ = image.shape
    n, _ = start_points.shape

    image = image.copy()
    start_cols = start_points[:, 0]
    start_rows = start_points[:, 1]
    end_cols = end_points[:, 0]
    end_rows = end_points[:, 1]

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


def _draw_ellipses(
    image: ndarray,
    start_points: ndarray,
    end_points: ndarray,
    colors: ndarray,
    thickness: int = 1,
) -> ndarray:
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

    h, w, _ = image.shape
    n, _ = start_points.shape

    image = image.copy()
    start_cols = start_points[:, 0]
    start_rows = start_points[:, 1]
    end_cols = end_points[:, 0]
    end_rows = end_points[:, 1]

    for i in range(n):
        sr, er = start_rows[i], end_rows[i]
        sc, ec = start_cols[i], end_cols[i]
        rr, cc = draw.ellipse(
            r=(sr + er) / 2,
            c=(sc + ec) / 2,
            r_radius=math.sqrt((sr - er) ** 2 + (sc - ec) ** 2) / 2,
            c_radius=thickness,
            shape=(h, w),
            rotation=math.atan2(ec - sc, er - sr),
        )
        image[rr, cc] = colors[i]

    return image


def draw_lines(
    image: ndarray,
    start_points: ndarray,
    end_points: ndarray,
    colors: Optional[ndarray] = None,
    thickness: int = 1,
    mode: str = "line",
) -> ndarray:
    """

    :param image: [*, H, W, C]
    :param start_points: [*, N, 2]
    :param end_points: [*, N, 2]
    :param colors: [*, N, C]
    :param thickness: [*, N, C]
    :param mode: "line" or "ellipse"
    :return: [*, H, W, C]
    """
    assert image.shape[:-3] == start_points.shape[:-2] == end_points.shape[:-2]
    assert start_points.shape[-1] == end_points.shape[-1] == 2
    assert mode in ["line", "ellipse"]

    if mode in "line":
        _draw_func = _draw_lines
    else:
        _draw_func = _draw_ellipses

    shape = image.shape
    dtype = image.dtype

    image = image.reshape((-1,) + image.shape[-3:])
    start_points = start_points.reshape((-1,) + start_points.shape[-2:])
    end_points = end_points.reshape((-1,) + end_points.shape[-2:])
    start_points = start_points.round().astype(dtype=np.int64)
    end_points = end_points.round().astype(dtype=np.int64)

    b, _, _, c = image.shape
    _, n, _ = start_points.shape

    if colors is None:
        colors = get_palette(n, dtype=dtype)

    assert start_points.shape[-2] == end_points.shape[-2] == colors.shape[-2]
    assert image.shape[-1] == colors.shape[-1]

    for i in range(b):
        image[i] = _draw_func(
                image=image[i],
                start_points=start_points[i],
                end_points=end_points[i],
                colors=colors,
                thickness=thickness,
            )

    image = image.reshape(shape)

    return image


def draw_pose(
    image: ndarray,
    points: ndarray,
    confidences: ndarray,
    edges: Tuple[Tuple[int, int], ...],
    circle_colors: Optional[ndarray] = None,
    circle_radius: int = 1,
    line_colors: Optional[ndarray] = None,
    line_thickness: int = 1,
) -> ndarray:
    """

    :param image: [*, H, W, C]
    :param points: [*, N, 2]
    :param confidences: [*, N, 1]
    :param edges: [M, 2]
    :param circle_colors: [N, C]
    :param circle_radius:
    :param line_colors: [M, C]
    :param line_thickness:
    :return: [*, H, W, C]
    """
    assert image.shape[:-3] == points.shape[:-2]
    assert points.shape[-1] == 2

    if line_thickness > 0:
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
            thickness=line_thickness,
            mode="line",
        )

    if circle_radius > 0:
        points = points.copy()
        confidences = np.concatenate([confidences, confidences], axis=-1)
        points[~confidences] = -1.0

        image = draw_points(
            image=image,
            points=points,
            colors=circle_colors,
            radius=circle_radius,
            mode="circle",
        )

    return image
