import re
from typing import Any, Callable


def compile_query(path: str) -> Callable[[dict | object], Any]:
    """Compiles a function to extract the sub-object at path from a json object.

    The expression is a dot separated set of components, such as:
        x.y.z

    Example::
        query = compile_query('a.b.c')
        query({'a': {'b': {'c': 10}}})
        => 10

    Query supports '*' (for all) and '?' (for any) as in:
        a.*.c 
    Which will return a list of all matching members as a tuple.

    Args:
        path (str): A json path expression, see above.

    Returns:
        Callable[[dict | object], Any]: The compiled function to extract the json member.
    """
    keys = path.split(".")

    def query(record: dict | object) -> Any:
        val = record
        for i, key in enumerate(keys):
            match val:
                case None:
                    return None
                case list() if (key == '*' or key == '?'):
                    rest = ".".join(keys[i+1:])
                    if not rest:
                        return (key, val) if isinstance(val, list) else (key, list(val))
                    else:
                        sub_query = compile_query(rest)
                        return key, [sub_query(item) for item in val]
                case list():
                    val = val[int(key)]
                case dict():
                    val = val.get(key)
                case _:
                    val = getattr(val, key, None)
        return val
    return query


_FILTER_RE = re.compile(
    r"^\s*(?P<path>[\w.+*?]+)\s*(?P<op>==|!=|>=|<=|>|<)\s*(?P<value>['\"].*?['\"]|-?\d+)\s*$")


def compile_filter(expression: str) -> Callable[[dict | object], bool]:
    if (m := _FILTER_RE.match(expression)) is None:
        raise ValueError(
            f"No valid operator found in expression: {expression!r}")
    path, op, value = m.group('path'), m.group('op'), m.group('value')
    query = compile_query(path)
    # parse the literal type
    if value.startswith("'") or value.startswith('"'):
        value = value.strip("'\"")
        def cmp(a, v=value): return a == v
    else:
        value = int(value)
        match op:
            case "==": cmp = lambda a, v=value: a == v
            case "!=": cmp = lambda a, v=value: a != v
            case ">=": cmp = lambda a, v=value: a >= v
            case "<=": cmp = lambda a, v=value: a <= v
            case ">": cmp = lambda a, v=value: a > v
            case "<": cmp = lambda a, v=value: a < v

    def make_filter():
        def f(record):
            result = query(record)
            match result:
                case None:
                    return False
                case ('?', list() as items):
                    return any(x is not None and cmp(x) for x in items)
                case ('*', list() as items):
                    if items:
                        return all(x is not None and cmp(x) for x in items)
                    else:
                        return False
                case _:
                    return cmp(result)
        return f
    return make_filter()
