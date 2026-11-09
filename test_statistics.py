"""Verify the assignment's percentage-point calculations and missing-data rule."""
import unittest
from area_statistics import summarize


class StatisticsTests(unittest.TestCase):
    def test_changes_and_totals(self):
        maps = [[0] * (20 + i) + [1] * (80 - i) for i in range(12)]
        result = summarize(list(range(12)), maps, {0: 'A', 1: 'B'}, 100)
        a = [row for row in result['rows'] if row['class_id'] == 0]
        self.assertEqual(a[0]['percentage'], 20)
        self.assertEqual(a[-1]['percentage'], 31)
        self.assertIsNone(a[0]['monthly_change_pp'])
        self.assertAlmostEqual(sum(row['monthly_change_pp'] for row in a[1:]), 11)
        self.assertEqual(a[-1]['change_since_first_pp'], 11)
        for month in range(12):
            self.assertAlmostEqual(sum(row['percentage'] for row in result['rows']
                                       if row['month'] == month), 100)

    def test_common_valid_footprint(self):
        maps = [[0, 1, 1] for _ in range(12)]
        maps[-1][2] = None
        result = summarize(list(range(12)), maps, {0: 'A', 1: 'B'}, 100)
        self.assertEqual(result['valid_pixels'], 2)
        self.assertTrue(all(row['percentage'] == 50 for row in result['rows']))

    def test_invalid_inputs(self):
        for maps in ([[0]] * 11, [[None]] * 12, [[9]] * 12):
            with self.assertRaises(ValueError):
                summarize(list(range(12)), maps, {0: 'A'}, 100)


if __name__ == '__main__':
    unittest.main()
