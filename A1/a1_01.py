import cv2
import pandas as pd
import numpy as np
import os
def read_image(filename : str)-> np.array:
    return cv2.imread(filename)
img = read_image("data/cse-logo.png")
print(img.shape)
print(img)
