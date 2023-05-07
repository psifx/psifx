from typing import Union

import shlex
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import time

import pandas as pd
import numpy as np

from psifx.video.face.tool import FaceAnalysisTool
from psifx.video.face.openface.fields import CLEAN_FIELDS, DIRTY_FIELDS
from psifx.io import tar

EXECUTABLE_PATH = Path(shutil.which("FeatureExtraction")).resolve(strict=True)
DEFAULT_OPTIONS = "-2Dfp -3Dfp -pdmparams -pose -aus -gaze -au_static"


class OpenFaceAnalysisTool(FaceAnalysisTool):
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
        video_path = Path(video_path)
        features_path = Path(features_path)

        if self.verbose:
            print(f"video       =   {video_path}")
            print(f"features    =   {features_path}")

        assert video_path.is_file()

        tmp_dir = Path(f"/tmp/TEMP_{time.time()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        args = f"{EXECUTABLE_PATH} -f {video_path} -out_dir {tmp_dir} {DEFAULT_OPTIONS}"

        if self.verbose:
            print("OpenFace will run with the following command:")
            print(f"{args}")
            print("It might take a while, depending on the length of the video and the")
            print("number of CPUs.")

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
        for clean, dirty in zip(CLEAN_FIELDS, DIRTY_FIELDS):
            clean_dataframe[clean] = dirty_dataframe[dirty].values.tolist()

        n_rows, n_cols = clean_dataframe.shape
        features = {}
        for i in tqdm(
            range(n_rows),
            desc="Parsing",
            disable=not self.verbose,
        ):
            index = clean_dataframe["index"][i]
            features[f"{index: 015d}"] = {
                clean: np.array(clean_dataframe[clean][i]).flatten().tolist()
                for clean in CLEAN_FIELDS
            }

        shutil.rmtree(tmp_dir)

        dictionary = {
            f"{k}.json": json.dumps(v)
            for k, v in tqdm(
                features.items(),
                desc="Encoding",
                disable=not self.verbose,
            )
        }
        tar.dump(
            dictionary=dictionary,
            path=features_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

    def visualization(
        self,
        video_path: Union[str, Path],
        features_path: Union[str, Path],
        visualization_path: Union[str, Path],
    ):
        raise NotImplementedError


def inference_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing files, otherwise raises an error.",
    )
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Verbosity of the script.",
    )
    args = parser.parse_args()

    tool = OpenFaceAnalysisTool(
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.inference(
        video_path=args.video,
        features_path=args.features,
    )
    del tool
