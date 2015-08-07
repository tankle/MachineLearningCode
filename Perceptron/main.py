# -*- coding: utf-8 -*-
__author__ = 'tan'

import numpy as np


def func(x, w, b):
    return np.dot(w,x) + b

def train(x,y,pha):
    n, m = np.shape(x)

    # w = np.array([np.random.uniform(0, 1) for _ in range(m)])
    # b = np.random.uniform(0,1)
    w = np.array([0 for _ in range(m)])
    b = 0

    print(w)
    print(b)

    flag = True
    idx = 1
    while flag:
        idx += 1
        if idx % 10 == 0:
            print("Iteration %d" % idx)
        flag = False
        for i in range(n):
            res = y[i] * func(x[i], w, b)
            if res <= 0:
                flag = True
                w = w + pha*y[i]*x[i]
                b = b + pha*y[i]
    return w, b

if __name__ == "__main__":
    # print(np.random(1))
    x = np.array([[3,3],[4,3],[1,1]])
    y = np.array([1,1,-1])

    print(train(x,y,1))