"""Misc utilities, too small to live in ther own module."""

import logging
import os
import subprocess
import sys
from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Union, cast, get_args, get_origin

import torch

DeviceType = Union[str, torch.device]


def log_uncaught_exceptions() -> None:
    """Route uncaught exceptions through ``logging`` instead of the default hook.

    Without this, an unhandled exception is printed by Python's default
    ``sys.excepthook`` to stderr, bypassing ``logging`` entirely — so it lands
    in the terminal even when ``--log-file`` redirected logging to a file. This
    hook logs the traceback as CRITICAL, so it follows the configured logging
    destination. ``KeyboardInterrupt`` is left to the default hook so Ctrl-C
    stays clean.
    """

    def handler(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = handler


def iterable_from_file(path: str | Path) -> Iterable[str]:
    with open(path, "r") as file:
        yield from (line.rstrip() for line in file)


def current_commit() -> str:
    """Get the last Git commit hash."""
    try:
        # Use '--short' for a shorter hash if needed
        args = ["git", "rev-parse", "HEAD"]

        # Run the Git command and decode the output
        hash = subprocess.check_output(args).strip().decode("utf-8")
        return hash
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving commit hash: {e}")
        return "unknown-commit"


def path_substract(shorter: Path, longer: Path) -> Path:
    """Substract the shorter path from the longer to obtain a relative path.

        This function asserts that there is a common prefix to both paths.

    Args:
        shorter (Path): The short path to remove.
        longer (Path): The longer path to remove shorter from.

    Returns:
        Path: _description_
    """
    prefix = os.path.commonprefix([shorter, longer])
    assert prefix is not None, f"Can't substract {shorter} from {longer}"
    return Path(os.path.relpath(longer, prefix))


def from_json(cls: type, data: Any) -> Any:
    """Deserializes a json object into a dataclass instance.

    This handles nested data classes and dict and list fields.

    Args:
        cls (type): Target class.
        data (Any): The json object to deserialize.

    Returns:
        _type_: An instance of the target class.
    """
    if isinstance(cls, type) and issubclass(cls, Enum):
        return cls(data)

    if is_dataclass(cls):
        field_types = {f.name: f.type for f in fields(cls)}
        return cls(
            **{
                key: from_json(cast(type, field_types[key]), value)
                for key, value in data.items()
            }
        )

    origin = get_origin(cls)

    if origin is list:
        item_type = get_args(cls)[0]
        return [from_json(item_type, item) for item in data]
    elif origin is dict:
        item_type = get_args(cls)[1]
        return {key: from_json(item_type, value) for key, value in data.items()}
    else:
        return data


def print_histogram(counter: Counter, title: str, width: int = 80) -> None:
    print(title)
    max_val = max(counter.values())
    total = sum(counter.values())
    cover = 0.0
    for key, count in sorted(counter.items()):
        bar = "█" * int(count / max_val * width)
        cover += count / total
        print(f"{key:4d} | {bar} {count:,} {cover:.1%}")
