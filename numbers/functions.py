import matplotlib.pyplot as plt
import numpy as np
import os

def euclidianDistance(x, y, N)
    x = x.reshape(N, 1)
    y = y.reshape(N, 1) #reshapes both vectors
    return ((x-y).T).dot(x-y) #returns dot product