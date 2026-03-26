from typing import Any, Callable


def compile_query(path: str) -> Callable[[dict], Any]:
    keys = path.split(".")

    def query(record: dict) -> Any:
        val = record
        for key in keys:
            if val is None:
                return None
            val = val[int(key)] if isinstance(val, list) else val.get(key)
        return val
    return query
