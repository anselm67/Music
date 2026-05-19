import unittest

from utils import compile_filter, compile_query


class TestCompileQuery(unittest.TestCase):
    def test_simple_key(self) -> None:
        q = compile_query("name")
        self.assertEqual(q({"name": "Alice"}), "Alice")

    def test_nested_key(self) -> None:
        q = compile_query("address.city")
        self.assertEqual(q({"address": {"city": "Paris"}}), "Paris")

    def test_deeply_nested(self) -> None:
        q = compile_query("a.b.c.d")
        self.assertEqual(q({"a": {"b": {"c": {"d": 42}}}}), 42)

    def test_missing_key_returns_none(self) -> None:
        q = compile_query("address.city")
        self.assertIsNone(q({"address": {}}))

    def test_missing_intermediate_key_returns_none(self) -> None:
        q = compile_query("address.city")
        self.assertIsNone(q({}))

    def test_none_propagation(self) -> None:
        q = compile_query("a.b.c")
        self.assertIsNone(q({"a": None}))

    def test_list_index(self) -> None:
        q = compile_query("people.0.name")
        self.assertEqual(q({"people": [{"name": "Alice"}, {"name": "Bob"}]}), "Alice")

    def test_list_index_last(self) -> None:
        q = compile_query("people.1.name")
        self.assertEqual(q({"people": [{"name": "Alice"}, {"name": "Bob"}]}), "Bob")

    def test_integer_value(self) -> None:
        q = compile_query("age")
        self.assertEqual(q({"age": 30}), 30)

    def test_boolean_value(self) -> None:
        q = compile_query("active")
        self.assertIs(q({"active": False}), False)

    def test_compile_once_reuse(self) -> None:
        q = compile_query("name")
        records = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        self.assertEqual([q(r) for r in records], ["Alice", "Bob", "Carol"])


class TestCompileFilter(unittest.TestCase):
    def test_str_equality_match(self) -> None:
        f = compile_filter("name == 'Alice'")
        self.assertTrue(f({"name": "Alice"}))

    def test_str_equality_no_match(self) -> None:
        f = compile_filter("name == 'Alice'")
        self.assertFalse(f({"name": "Bob"}))

    def test_str_double_quotes(self) -> None:
        f = compile_filter('name == "Alice"')
        self.assertTrue(f({"name": "Alice"}))

    def test_int_equality(self) -> None:
        f = compile_filter("age == 30")
        self.assertTrue(f({"age": 30}))

    def test_int_not_equal(self) -> None:
        f = compile_filter("age != 30")
        self.assertTrue(f({"age": 25}))
        self.assertFalse(f({"age": 30}))

    def test_int_greater_than(self) -> None:
        f = compile_filter("age > 28")
        self.assertTrue(f({"age": 30}))
        self.assertFalse(f({"age": 28}))

    def test_int_greater_than_or_equal(self) -> None:
        f = compile_filter("age >= 30")
        self.assertTrue(f({"age": 30}))
        self.assertFalse(f({"age": 29}))

    def test_int_less_than(self) -> None:
        f = compile_filter("age < 30")
        self.assertTrue(f({"age": 25}))
        self.assertFalse(f({"age": 30}))

    def test_int_less_than_or_equal(self) -> None:
        f = compile_filter("age <= 30")
        self.assertTrue(f({"age": 30}))
        self.assertFalse(f({"age": 31}))

    def test_nested_path(self) -> None:
        f = compile_filter("address.city == 'Paris'")
        self.assertTrue(f({"address": {"city": "Paris"}}))
        self.assertFalse(f({"address": {"city": "London"}}))

    def test_missing_value_does_not_raise(self) -> None:
        f = compile_filter("age > 28")
        self.assertFalse(f({}))

    def test_invalid_expression_raises(self) -> None:
        with self.assertRaises(ValueError):
            compile_filter("foo.bar.baz")

    def test_compile_once_reuse(self) -> None:
        f = compile_filter("age > 28")
        records = [{"age": 25}, {"age": 30}, {"age": 35}]
        self.assertEqual([r for r in records if f(r)], [{"age": 30}, {"age": 35}])


class TestCompileQueryWildcard(unittest.TestCase):
    def test_wildcard_simple(self) -> None:
        q = compile_query("pages.*")
        self.assertEqual(q({"pages": [1, 2, 3]}), ("*", [1, 2, 3]))

    def test_wildcard_nested(self) -> None:
        q = compile_query("pages.*.staff_count")
        data = {"pages": [{"staff_count": 10}, {"staff_count": 40}]}
        self.assertEqual(q(data), ("*", [10, 40]))

    def test_wildcard_not_a_list_returns_none(self) -> None:
        q = compile_query("pages.*.staff_count")
        self.assertIsNone(q({"pages": "oops"}))

    def test_wildcard_empty_list(self) -> None:
        q = compile_query("pages.*.staff_count")
        self.assertEqual(q({"pages": []}), ("*", []))

    def test_wildcard_missing_key_in_items(self) -> None:
        q = compile_query("pages.*.staff_count")
        data = {"pages": [{"staff_count": 10}, {}]}
        self.assertEqual(q(data), ("*", [10, None]))


class TestCompileFilterWildcard(unittest.TestCase):
    def test_any_match(self) -> None:
        f = compile_filter("pages.?.staff_count > 30")
        data = {"pages": [{"staff_count": 10}, {"staff_count": 40}]}
        self.assertTrue(f(data))

    def test_all_match(self) -> None:
        f = compile_filter("pages.*.staff_count > 30")
        data = {"pages": [{"staff_count": 10}, {"staff_count": 40}]}
        self.assertFalse(f(data))
        f = compile_filter("pages.*.staff_count > 5")
        self.assertTrue(f(data))

    def test_no_match(self) -> None:
        f = compile_filter("pages.*.staff_count > 30")
        data = {"pages": [{"staff_count": 10}, {"staff_count": 20}]}
        self.assertFalse(f(data))

    def test_empty_list(self) -> None:
        f = compile_filter("pages.*.staff_count > 30")
        self.assertFalse(f({"pages": []}))

    def test_none_items_skipped(self) -> None:
        f = compile_filter("pages.*.staff_count > 30")
        # second has no staff_count
        data = {"pages": [{"staff_count": 40}, {}]}
        self.assertFalse(f(data))

        f = compile_filter("pages.?.staff_count > 30")
        # second has no staff_count
        data = {"pages": [{"staff_count": 40}, {}]}
        self.assertTrue(f(data))  # still True because first matches

    def test_all_none_items(self) -> None:
        f = compile_filter("pages.*.staff_count > 30")
        data: dict[str, object] = {"pages": [{}, {}]}  # no staff_count anywhere
        self.assertFalse(f(data))

    def test_not_a_list_returns_false(self) -> None:
        f = compile_filter("pages.*.staff_count > 30")
        self.assertFalse(f({"pages": "oops"}))

    def test_wildcard_str_equality(self) -> None:
        f = compile_filter("pages.?.clef == 'treble'")
        data = {"pages": [{"clef": "bass"}, {"clef": "treble"}]}
        self.assertTrue(f(data))


if __name__ == "__main__":
    unittest.main()
