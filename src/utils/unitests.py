# todo:
#   check that there is not any problem with the dimensions

import unittest
from hBeta_fast import perm_matrix, med_vec_array
import numpy as np
import collections

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

class Perm(unittest.TestCase):

    def test_2d_perm(self):
        a = perm_matrix(p=2, len_1_dim=4)
        arr_ind = np.array([[i, j] for i in range(4) for j in range(4)])
        self.assertEqual(np.sum(a[:, 0]), np.sum(arr_ind[:, 0]))
        self.assertEqual(np.sum(arr_ind == a), 2 * 4**2)

    def test_3d_perm(self):
        a = perm_matrix(p=3, len_1_dim=3)
        arr_ind = np.array([[i, j, k] for i in range(3) for j in range(3)
                            for k in range(3)])
        self.assertEqual(np.sum(a[:, 0]), np.sum(arr_ind[:, 0]))
        self.assertEqual(np.sum(arr_ind == a), 3 * 3**3)

    def test_med_vec_array(self):
        dict_of_seg = collections.defaultdict(dict)
        dict_of_seg = {0: {'vec': np.array([0., 0.25, 0.5, 0.75, 1.]),
                           'med_vec': np.array([-0.125, 0.125, 0.375, 0.625, 0.875]), 'diff': 0.25, 'min': 0, 'max': 1},
                       1: {'vec': np.array([0., 0.25, 0.5, 0.75, 1.]),
                           'med_vec': np.array([-0.125, 0.125, 0.375, 0.625, 0.875]), 'diff': 0.25, 'min': 0, 'max': 1}}
        arr_ind = perm_matrix(p=2, len_1_dim=4)
        arr_med = np.array(
            [[dict_of_seg[0]['med_vec'][i+1], dict_of_seg[1]['med_vec'][j+1]] for i in range(4)
             for j in range(4)])
        arr_vec = np.array(
            [[dict_of_seg[0]['vec'][i], dict_of_seg[1]['vec'][j]] for i in range(4)
             for j in range(4)])
        a_med, a_vec = med_vec_array(dict_of_seg, arr_ind)
        self.assertEqual(np.sum(arr_med == a_med), 2 * 16)
        self.assertEqual(np.sum(arr_vec == a_vec), 2 * 16)

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()