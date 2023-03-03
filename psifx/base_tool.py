from typing import Any, Union


class BaseTool(object):
    def __init__(
        self,
        device: str,
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        self.device = device
        self.overwrite = overwrite
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
