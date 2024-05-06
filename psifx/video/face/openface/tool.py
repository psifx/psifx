"""OpenFace face analysis tool."""

from typing import Optional, Union

import shlex
import shutil
import subprocess
from pathlib import Path
from PIL import Image
import json
import time
from tqdm import tqdm

import pandas as pd
import numpy as np
from numpy import ndarray

from psifx.video.face.tool import FaceAnalysisTool
from psifx.video.face.openface import skeleton, fields
from psifx.io import tar, video
from psifx.utils import draw

EXECUTABLE_PATH = shutil.which("FeatureExtraction")
if EXECUTABLE_PATH is not None:
    EXECUTABLE_PATH = Path(EXECUTABLE_PATH).resolve(strict=True)
DEFAULT_OPTIONS = "-2Dfp -3Dfp -pdmparams -pose -aus -gaze -au_static"


def gaze_vector_2d(
    eye_2d: ndarray,
    gaze_3d,
    depth: float,
    K: ndarray,
    K_inverse: ndarray,
):
    """
    Projects the gaze vector in 2D starting from the center of the eye.

    :param eye_2d:
    :param gaze_3d:
    :param depth:
    :param K:
    :param K_inverse:
    :return:
    """
    eye_center_2d = eye_2d[[21, 23, 25, 27]].mean(axis=-2)
    eye_center_2d = np.concatenate([eye_center_2d, np.ones(1)])
    eye_center_3d = depth * K_inverse @ eye_center_2d
    gaze_keypoint_3d = eye_center_3d + 0.1 * gaze_3d
    gaze_keypoint_2d = K @ (gaze_keypoint_3d / np.maximum(gaze_keypoint_3d[-1], 1e-8))
    return eye_center_2d[:-1], gaze_keypoint_2d[:-1]


class OpenFaceTool(FaceAnalysisTool):
    """
    OpenFace face analysis tool.

    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
        self,
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device="cpu",
            overwrite=overwrite,
            verbose=verbose,
        )

    def inference(
        self,
        video_path: Union[str, Path],
        features_path: Union[str, Path],
    ):
        """
        Implementation of OpenFace's face analysis inference method.

        :param video_path: The path to the video file.
        :param features_path: The path to the features archive.
        :return:
        """
        video_path = Path(video_path)
        features_path = Path(features_path)

        if self.verbose:
            print(f"video       =   {video_path}")
            print(f"features    =   {features_path}")

        assert video_path.is_file()
        tar.TarWriter.check(path=features_path, overwrite=self.overwrite)

        tmp_dir = Path(f"/tmp/TEMP_{time.time()}")
        tmp_dir.mkdir(parents=True)

        args = f"{EXECUTABLE_PATH} -f {video_path} -out_dir {tmp_dir} {DEFAULT_OPTIONS}"

        if self.verbose:
            print("OpenFace will run with the following command:")
            print(f"{args}")
            print("It might take a while, depending on the number of CPUs.")

        try:
            for i in tqdm(
                range(1),
                desc="Processing",
                disable=not self.verbose,
            ):
                process = subprocess.run(
                    args=shlex.split(args),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            if self.verbose > 1:
                print(process.stdout)

        except subprocess.CalledProcessError as error:
            print(error.stdout)

        dirty_dataframe = pd.read_csv(tmp_dir / (video_path.stem + ".csv"))

        clean_dataframe = pd.DataFrame()
        clean_dataframe["index"] = dirty_dataframe["frame"] - 1
        for clean, dirty in zip(fields.CLEAN_FIELDS, fields.DIRTY_FIELDS):
            clean_dataframe[clean] = dirty_dataframe[dirty].values.tolist()

        n_rows, _ = clean_dataframe.shape
        features = {
            "edges": {
                "eye_right_keypoints_2d": skeleton.EYE_EDGES,
                "eye_left_keypoints_2d": skeleton.EYE_EDGES,
                "eye_right_keypoints_3d": skeleton.EYE_EDGES,
                "eye_left_keypoints_3d": skeleton.EYE_EDGES,
                "face_keypoints_2d": skeleton.FACE_EDGES,
                "face_keypoints_3d": skeleton.FACE_EDGES,
            }
        }
        for i in tqdm(
            range(n_rows),
            desc="Parsing",
            disable=not self.verbose,
        ):
            index = clean_dataframe["index"][i]
            features[f"{index: 015d}"] = {
                field: np.array(clean_dataframe[field][i]).flatten().tolist()
                for field in fields.CLEAN_FIELDS
            }

        shutil.rmtree(tmp_dir)

        features = {
            f"{k}.json": json.dumps(v)
            for k, v in tqdm(
                features.items(),
                desc="Encoding",
                disable=not self.verbose,
            )
        }
        tar.TarWriter.write(
            dictionary=features,
            path=features_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

    def visualization(
        self,
        video_path: Union[str, Path],
        features_path: Union[str, Path],
        visualization_path: Union[str, Path],
        depth: Optional[float] = 3.0,
        f_x: Optional[float] = None,
        f_y: Optional[float] = None,
        c_x: Optional[float] = None,
        c_y: Optional[float] = None,
    ):
        """
        Produces a visualization of the face pose and eye gaze vectors over a video.

        :param video_path: The path to the video file.
        :param features_path: The path to the features archive.
        :param visualization_path: The path to the visualization file.
        :param depth: The (guesstimated) depth between the camera and the subject.
        :param f_x: The (guesstimated) focal length of the x-axis.
        :param f_y: The (guesstimated) focal length of the y-axis.
        :param c_x: The (guesstimated) principal point of the x-axis.
        :param c_y: The (guesstimated) principal point of the x-axis.
        :return:
        """
        video_path = Path(video_path)
        features_path = Path(features_path)
        visualization_path = Path(visualization_path)

        if self.verbose:
            print(f"video           =   {video_path}")
            print(f"features        =   {features_path}")
            print(f"visualization   =   {visualization_path}")

        assert video_path != visualization_path
        tar.TarReader.check(path=features_path)

        calibration = all(p is not None for p in [f_x, f_y, c_x, c_y])
        no_calibration = all(p is None for p in [f_x, f_y, c_x, c_y])
        assert calibration or no_calibration

        features = tar.TarReader.read(
            features_path,
            verbose=self.verbose,
        )

        try:
            edges = features.pop("edges.json")
            edges = json.loads(edges)
            edges = {k: tuple(v) for k, v in edges.items()}
        except KeyError:
            print("Missing or incorrect edges.json, only the landmarks will be drawn.")
            pose = next(iter(features.values()))
            pose = json.loads(pose)
            edges = {key: () for key, value in pose.items()}

        features = {
            int(k.replace(".json", "")): json.loads(v)
            for k, v in tqdm(
                features.items(),
                desc="Decoding",
                disable=not self.verbose,
            )
        }
        h, w = None, None
        K, K_inverse = None, None
        with (
            video.VideoReader(path=video_path) as video_reader,
            video.VideoWriter(
                path=visualization_path,
                input_dict={"-r": video_reader.frame_rate},
                output_dict={
                    "-c:v": "libx264",
                    "-crf": "15",
                    "-pix_fmt": "yuv420p",
                },
                overwrite=self.overwrite,
            ) as visualization_writer,
        ):
            for image, feature in zip(
                tqdm(
                    video_reader,
                    desc="Processing",
                    disable=not self.verbose,
                ),
                features.values(),
            ):
                h_, w_, _ = image.shape
                image = Image.fromarray(image.copy())
                for key in [
                    "face_keypoints_2d",
                    "eye_right_keypoints_2d",
                    "eye_left_keypoints_2d",
                ]:
                    points = np.array(feature[key]).reshape(-1, 2)
                    image = draw.draw_pose(
                        image=image,
                        points=points,
                        edges=edges[key],
                        circle_radius=0,
                        circle_thickness=0,
                        line_thickness=1,
                    )

                if h != h_ or w != w_:
                    h, w = h_, w_

                    if no_calibration:
                        f_x = 1600.0 / 1920.0 * w
                        f_y = 1600.0 / 1080.0 * h
                        c_x = w / 2
                        c_y = h / 2

                    K = np.array(
                        [
                            [f_x, 0.0, c_x],
                            [0.0, f_y, c_y],
                            [0.0, 0.0, 1.0],
                        ]
                    )
                    K_inverse = np.linalg.inv(K)

                center_right, gaze_right = gaze_vector_2d(
                    eye_2d=np.array(feature["eye_right_keypoints_2d"]).reshape(-1, 2),
                    gaze_3d=np.array(feature["gaze_right_3d"]),
                    depth=depth,
                    K=K,
                    K_inverse=K_inverse,
                )

                center_left, gaze_left = gaze_vector_2d(
                    eye_2d=np.array(feature["eye_left_keypoints_2d"]).reshape(-1, 2),
                    gaze_3d=np.array(feature["gaze_left_3d"]),
                    depth=depth,
                    K=K,
                    K_inverse=K_inverse,
                )

                image = draw.draw_pose(
                    image=image,
                    points=np.stack([center_right, gaze_right, center_left, gaze_left]),
                    edges=((0, 1), (2, 3)),
                    circle_radius=1,
                    circle_thickness=1,
                    line_thickness=1,
                )
                image = np.asarray(image)
                visualization_writer.write(image=image)
