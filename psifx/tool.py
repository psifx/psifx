from typing import Union


class BaseTool:
    """
    Base class for any tool.
    """

    def __init__(
        self,
        device: str,
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        self.device = device
        self.overwrite = overwrite
        self.verbose = verbose
