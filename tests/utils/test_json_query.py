import unittest


def compile_query(path: str):
    keys = path.split(".")
    def query(record):
        val = record
        for key in keys:
            if val is None:
                return None
            val = val[int(key)] if isinstance(val, list) else val.get(key)
        return val
    return query


class TestCompileQuery(unittest.TestCase):

    def test_simple_key(self):
        q = compile_query("name")
        self.assertEqual(q({"name": "Alice"}), "Alice")

    def test_nested_key(self):
        q = compile_query("address.city")
        self.assertEqual(q({"address": {"city": "Paris"}}), "Paris")

    def test_deeply_nested(self):
        q = compile_query("a.b.c.d")
        self.assertEqual(q({"a": {"b": {"c": {"d": 42}}}}), 42)

    def test_missing_key_returns_none(self):
        q = compile_query("address.city")
        self.assertIsNone(q({"address": {}}))

    def test_missing_intermediate_key_returns_none(self):
        q = compile_query("address.city")
        self.assertIsNone(q({}))

    def test_none_propagation(self):
        q = compile_query("a.b.c")
        self.assertIsNone(q({"a": None}))

    def test_list_index(self):
        q = compile_query("people.0.name")
        self.assertEqual(q({"people": [{"name": "Alice"}, {"name": "Bob"}]}), "Alice")

    def test_list_index_last(self):
        q = compile_query("people.1.name")
        self.assertEqual(q({"people": [{"name": "Alice"}, {"name": "Bob"}]}), "Bob")

    def test_integer_value(self):
        q = compile_query("age")
        self.assertEqual(q({"age": 30}), 30)

    def test_boolean_value(self):
        q = compile_query("active")
        self.assertIs(q({"active": False}), False)

    def test_compile_once_reuse(self):
        q = compile_query("name")
        records = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        self.assertEqual([q(r) for r in records], ["Alice", "Bob", "Carol"])


if __name__ == "__main__":
    unittest.main()