"""drawing utilities."""

from typing import Optional, Tuple

from colorsys import hls_to_rgb

import numpy as np
from PIL import Image, ImageDraw


def get_palette(
    num_colors: int,
    hue: float = 0.01,
    luminance: float = 0.6,
    saturation: float = 0.65,
) -> np.ndarray:
    """
    Returns a palette of colours between [0, 1].

    :param num_colors:
    :param hue:
    :param luminance:
    :param saturation:
    :return: [N, 3]
    """
    hues = np.linspace(0.0, 1.0, num_colors + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(dtype=np.int64)
    hues = hues.tolist()
    palette = np.array([hls_to_rgb(hue, luminance, saturation) for hue in hues])
    palette *= 255.0
    return palette.astype(dtype=np.uint8)


def draw_points(
    image: Image.Image,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: int = 2,
    thickness: int = 1,
) -> Image.Image:
    """
    Draws points on an image.

    :param image:
    :param points: [N, 2]
    :param colors: [N, C]
    :param radius:
    :return:
    """
    if colors is None:
        colors = get_palette(points.shape[-2])

    assert points.shape[-2] == colors.shape[-2]
    assert points.shape[-1] == 2

    draw = ImageDraw.Draw(image)

    for point, color in zip(points, colors):
        draw.ellipse(
            xy=[tuple(point - radius), tuple(point + radius)],
            outline=tuple(color),
            fill=(255, 255, 255),
            width=thickness,
        )

    return image


def draw_lines(
    image: Image.Image,
    start_points: np.ndarray,
    end_points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    thickness: int = 3,
) -> Image.Image:
    """
    Draws lines on an image.

    :param image:
    :param start_points: [N, 2]
    :param end_points: [N, 2]
    :param colors: [N, C]
    :param thickness:
    :return:
    """
    if colors is None:
        colors = get_palette(start_points.shape[-2])

    assert start_points.shape[-2] == end_points.shape[-2] == colors.shape[-2]
    assert start_points.shape[-1] == end_points.shape[-1] == 2

    draw = ImageDraw.Draw(image)

    for start_point, end_point, color in zip(start_points, end_points, colors):
        draw.line(
            xy=[tuple(start_point), tuple(end_point)],
            fill=tuple(color),
            width=thickness,
        )

    return image


def draw_pose(
    image: Image.Image,
    points: np.ndarray,
    edges: Tuple[Tuple[int, int], ...],
    confidences: Optional[np.ndarray] = None,
    circle_colors: Optional[np.ndarray] = None,
    circle_radius: int = 2,
    circle_thickness: int = 1,
    line_colors: Optional[np.ndarray] = None,
    line_thickness: int = 3,
) -> Image.Image:
    """
    Draws a skeleton pose on an image.

    :param image: [H, W, C]
    :param points: [N, 2]
    :param confidences: [N, 1]
    :param edges: [M, 2]
    :param circle_colors: [N, C]
    :param circle_radius:
    :param circle_thickness:
    :param line_colors: [M, C]
    :param line_thickness:
    :return: [H, W, C]
    """
    if confidences is None:
        confidences = np.ones(points.shape[:-1] + (1,), dtype=np.bool_)

    assert points.ndim == confidences.ndim == 2
    assert points.shape[-2] == confidences.shape[-2]
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
            thickness=circle_thickness,
        )

    return image
