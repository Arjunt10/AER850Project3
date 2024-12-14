# -*- coding: utf-8 -*-
import cv2
import matplotlib as plt


" STEP 1: Object Masking " 

"Read Image"
image_orig = cv2.imread(r"motherboard_image.jpg")


"Rotate Image"
image_orig = cv2.rotate(image_orig,cv2.ROTATE_90_CLOCKWISE)

"Grayout Image"
image_gray = cv2.cvtColor(image_orig, cv2.COLOR_RGB2GRAY)

"Adaptive Threshold"
image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

"Edges & Image Dilation"
Edges = cv2.Canny (image_gray, 30, 100)

# Rectangular Kernel for Dilation
kernel = cv2.getStructureingElement (cv2.MORPH_Rect, (3,3))
Edges = cv2.dilate(Edges, kernel, iterations = 1)

"Contours"
contours, _ = cv2.findContours (Edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

"Mask"
mask = cv2.drawContours (cv2.zeros_like(image_gray), contours, -1, 255, thickness = cv2.FILLED)

"Apply Mask on Original Image"
image_final = cv2.bitwise_and (image_orig, image_orig, mask)

"Show image"
plt.show(image_final)

