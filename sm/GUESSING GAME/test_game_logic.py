import unittest
from game_logic import get_performance_rating

class TestPerformanceRating(unittest.TestCase):

    def test_excellent_rating(self):
        self.assertEqual(get_performance_rating(3), "Excellent")
        self.assertEqual(get_performance_rating(5), "Excellent")

    def test_very_good_rating(self):
        self.assertEqual(get_performance_rating(6), "Very Good")
        self.assertEqual(get_performance_rating(10), "Very Good")

    def test_average_rating(self):
        self.assertEqual(get_performance_rating(15), "Average")
        self.assertEqual(get_performance_rating(20), "Average")

    def test_needs_improvement_rating(self):
        self.assertEqual(get_performance_rating(21), "Needs Improvement")
        self.assertEqual(get_performance_rating(100), "Needs Improvement")

if __name__ == '__main__':
    unittest.main()
