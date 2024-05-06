"""``psifx`` base tool."""

from typing import Union


class Tool:
    """
    Base class for any tool.

    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
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
