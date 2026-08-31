import unittest

from scripts.query_budget import structural_queries, total_queries

#testing query
class QueryBudgetTests(unittest.TestCase):
    def test_one_d64_shared_stencil(self) -> None:
        self.assertEqual(structural_queries(d=64, probe_locations=1), 8193)

    def test_selected_total_budget(self) -> None:
        self.assertEqual(total_queries(d=64, probe_locations=1, completion_queries=12000), 20193)

    def test_unshared_t16_baseline(self) -> None:
        self.assertEqual(structural_queries(d=64, probe_locations=16), 131088)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            structural_queries(d=0, probe_locations=1)
        with self.assertRaises(ValueError):
            total_queries(d=64, probe_locations=1, completion_queries=-1)


if __name__ == "__main__":
    unittest.main()
