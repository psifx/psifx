from typing import Any, Dict, List, Union

import json
from pathlib import Path
from tqdm import tqdm


def load(
    path: Union[str, Path],
    verbose: Union[bool, int] = True,
) -> Union[List, Dict]:
    path = Path(path)

    assert path.suffix == ".json"

    with path.open(mode="r") as file:
        for i in tqdm(
            range(1),
            desc="Loading",
            disable=not verbose,
        ):
            data = json.load(file)

    return data


def dump(
    data: Union[List, Dict],
    path: Union[str, Path],
    overwrite: bool = False,
    verbose: Union[bool, int] = True,
):
    path = Path(path)

    assert path.suffix == ".json"

    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode="w") as file:
        for i in tqdm(range(1), desc="Dumping", disable=not verbose):
            json.dump(data, file)
