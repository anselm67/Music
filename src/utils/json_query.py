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


def compile_filter(expression: str) -> Callable[[dict], bool]:
    operators = ["==", "!=", ">=", "<=", ">", "<"]
    for op in operators:
        if op in expression:
            path, _, raw = expression.partition(op)
            query = compile_query(path.strip())
            literal = raw.strip()
            # parse the literal type
            if literal.startswith("'") or literal.startswith('"'):
                value = literal.strip("'\"")
                return lambda record, q=query, v=value: q(record) == v
            else:
                value = int(literal)
                match op:
                    case "==": return lambda record, q=query, v=value: (r := q(record)) is not None and r == v
                    case "!=": return lambda record, q=query, v=value: (r := q(record)) is not None and r != v
                    case ">=": return lambda record, q=query, v=value: (r := q(record)) is not None and r >= v
                    case "<=": return lambda record, q=query, v=value: (r := q(record)) is not None and r <= v
                    case ">": return lambda record, q=query, v=value: (r := q(record)) is not None and r > v
                    case "<": return lambda record, q=query, v=value: (r := q(record)) is not None and r < v

    raise ValueError(f"No valid operator found in expression: {expression!r}")
