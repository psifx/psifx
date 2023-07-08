from typing import Union

from pathlib import Path


class WAVReader(object):
    @staticmethod
    def check(path: Union[str, Path]):
        path = Path(path)
        if path.suffix != ".wav":
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)


class WAVWriter(object):
    @staticmethod
    def check(path: Union[str, Path]):
        path = Path(path)
        if path.suffix != ".wav":
            raise NameError(path)
