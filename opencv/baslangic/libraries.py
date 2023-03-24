from skimage import io
img1 = io.imread("ronaldo.png")

import cv2
img2 = cv2.imread("ronaldo.png")

import numpy as np
a = np.ones((5,5))

import pandas as pd #excel dosyası olucak

from matplotlib import pyplot as plt
plt.imshow(img1)