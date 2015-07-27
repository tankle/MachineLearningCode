#-*- coding: utf-8 -*-


from kmeans import *
import numpy as np
import unittest


class TestFactorial(unittest.TestCase):
    """
    Our basic test class
    """

    def test_dist(self):
        
        veca = np.array([1])
        vecb = np.array([2])
        res = distEclud(veca, vecb)
        self.assertEqual(res, 1)


if __name__ == '__main__':

#    unittest.main()
    dataset = mat(loadDataSet("testSet.txt"))
    mc,ca = kMeans(dataset, 4)
    print(mc,ca)
