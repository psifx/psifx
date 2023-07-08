from typing import Dict, List, Union

import json
from pathlib import Path
from tqdm import tqdm


class JSONReader(object):
    @staticmethod
    def check(path: Union[str, Path]):
        path = Path(path)
        if path.suffix != ".json":
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: Union[bool, int] = True,
    ) -> Union[List, Dict]:
        path = Path(path)
        JSONReader.check(path)

        with path.open(mode="r") as file:
            for i in tqdm(
                range(1),
                desc="Reading",
                disable=not verbose,
            ):
                data = json.load(file)

        return data


class JSONWriter(object):
    @staticmethod
    def check(path: Union[str, Path]):
        path = Path(path)
        if path.suffix != ".json":
            raise NameError(path)

    @staticmethod
    def write(
        data: Union[List, Dict],
        path: Union[str, Path],
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        path = Path(path)
        JSONWriter.check(path)

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="w") as file:
            for i in tqdm(range(1), desc="Writing", disable=not verbose):
                json.dump(data, file)
