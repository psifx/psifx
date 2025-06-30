"""video processing tool."""
from typing import Union

from psifx.tool import Tool


class VideoTool(Tool):
    """
    Base class for video tools.
    """
    def __init__(self, device: str, overwrite: bool = False, verbose: Union[bool, int] = True):
        super().__init__(device, overwrite, verbose)
        self.device = device
        self.overwrite = overwrite
        self.verbose = verbose
