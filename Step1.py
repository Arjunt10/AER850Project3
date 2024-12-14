# -*- coding: utf-8 -*-

import cv2
import numpy as np

" STEP 1: Object Masking " 

"Read Image"
image_orig = cv2.imread(r"motherboard_image.jpg")


"Rotate Image"
image_orig = cv2.rotate(image_orig,cv2.ROTATE_90_CLOCKWISE)

"Grayout Image"
image_gray = cv2.cvtColor(image_orig, cv2.COLOR_RGB2GRAY)

"Adaptive Threshold"
image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

"Edges"
Edges = cv2.Canny (image_gray, 50, 150)