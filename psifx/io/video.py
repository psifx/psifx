from typing import Dict, Optional, Union

from pathlib import Path

from numpy import ndarray

from skvideo.io import FFmpegReader, FFmpegWriter


class VideoReader(FFmpegReader):
    """
    Video reader object.
    """

    def __init__(
        self,
        path: Union[str, Path],
        input_dict: Optional[Dict[str, str]] = None,
        output_dict: Optional[Dict[str, str]] = None,
    ):
        path = Path(path)

        assert path.exists()

        super().__init__(
            filename=str(path),
            inputdict=input_dict,
            outputdict=output_dict,
        )

        self.num_frames = self.inputframenum
        self.frame_rate = self.probeInfo["video"][self.INFO_AVERAGE_FRAMERATE]

    def __len__(self):
        return self.num_frames


class VideoWriter(FFmpegWriter):
    """
    Video writer object.
    """

    def __init__(
        self,
        path: Union[str, Path],
        input_dict: Optional[Dict[str, str]] = None,
        output_dict: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ):
        path = Path(path)

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(path),
            inputdict=input_dict,
            outputdict=output_dict,
        )

    def write(self, image: ndarray):
        """
        Appends an image to the existing video
        :param image: [H, W, 3] ndarray.
        :return:
        """
        self.writeFrame(im=image)
