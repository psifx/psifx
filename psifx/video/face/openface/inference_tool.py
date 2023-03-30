from typing import Union

import shlex
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import time

import pandas as pd

from psifx.video.face.inference_tool import BaseFaceAnalysisTool
from psifx.video.face.openface.fields import CLEAN_FIELDS, DIRTY_FIELDS
from psifx.utils import tar, timestamp


class OpenFaceAnalysisTool(BaseFaceAnalysisTool):
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

    def __call__(
        self,
        video_path: Union[str, Path],
        features_path: Union[str, Path],
    ):
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        if not isinstance(features_path, Path):
            features_path = Path(features_path)

        assert video_path.is_file()

        tmp_dir = Path(f"/tmp/TEMP_{time.time()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        executable = "/home/guillaume/Projects/OpenFace/build/bin/FeatureExtraction"
        options = "-2Dfp -3Dfp -pdmparams -pose -aus -gaze -au_static"
        args = f"{executable} -f {video_path} -out_dir {tmp_dir} {options}"

        if self.verbose:
            message = (
                "OpenFace will run with the following command:\n"
                f"{args}\n"
                "It might take a while, depending on the length of the video and the number of CPUs."
            )
            print(message)

        # WEIRD BUT PRINTS IF NOT CAPTURED
        # process = subprocess.run(
        #     args=shlex.split(args),
        #     check=True,
        #     capture_output=not self.verbose > 1,
        #     text=True,
        # )

        # MORE CORRECT BUT LESS INTERACTIVE
        start = time.time()
        process = subprocess.run(
            args=shlex.split(args),
            check=True,
            stdout=subprocess.PIPE if self.verbose > 1 else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if self.verbose > 1 else subprocess.DEVNULL,
            text=True,
        )
        end = time.time()
        if self.verbose > 1:
            print(process.stdout)
        if self.verbose:
            print(f"OpenFace took {timestamp.format_timestamp(end - start)}.")

        # SLOW CODE WHERE WE ITERATE THROUGH ROWS OF THE DATAFRAME
        # df = pd.read_csv(tmp_dir / (video_path.stem + ".csv"))
        # features = {}
        # for i, row in tqdm(
        #     df.iterrows(),
        #     disable=not self.verbose,
        # ):
        #     features[f"{i: 015d}"] = {
        #         clean: row[dirty].to_numpy().flatten().tolist()
        #         for clean, dirty in zip(CLEAN_FIELDS, DIRTY_FIELDS)
        #     }

        # FAST CODE WHERE WE GROUP ALL THE COLUMNS INTO NUMPY ARRAYS
        # THEN WE PUT THEM IN THE INDEX-ORDERED DICT.
        df = pd.read_csv(tmp_dir / (video_path.stem + ".csv"))
        df2 = {}
        for clean, dirty in zip(CLEAN_FIELDS, DIRTY_FIELDS):
            df2[clean] = df[dirty].to_numpy()
        n_rows, n_cols = df.shape
        features = {}
        for i in tqdm(
            range(n_rows),
            disable=not self.verbose,
        ):
            features[f"{i: 015d}"] = {
                clean: df2[clean][i].flatten().tolist() for clean in CLEAN_FIELDS
            }

        shutil.rmtree(tmp_dir)

        if features_path.exists():
            if self.overwrite:
                features_path.unlink()
            else:
                raise FileExistsError(features_path)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        tar.dump(
            dictionary={f"{k}.json": json.dumps(v) for k, v in features.items()},
            path=features_path,
        )


def cli_main():
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

    if args.video.is_file():
        video_path = args.video
        features_path = args.features
        tool(
            video_path=video_path,
            features_path=features_path,
        )
    elif args.video.is_dir():
        video_dir = args.video
        features_dir = args.features
        for video_path in sorted(video_dir.glob("*.mp4")):
            features_name = video_path.stem + ".tar.gz"
            features_path = features_dir / features_name
            tool(
                video_path=video_path,
                features_path=features_path,
            )
    else:
        raise ValueError("args.video is neither a file or a directory.")

    del tool


if __name__ == "__main__":
    cli_main()
