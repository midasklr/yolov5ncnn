import os
import cv2
import numpy as np
im1 = cv2.imread("yolov5_ncnn_fp16.jpg")
im2 = cv2.imread("yolov5-int8-kl.jpg")
im3 = cv2.imread("yolov5-int8-aciq.jpg")

im1 = cv2.putText(im1, "NCNN FP16", (100,100), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 3)
im2 = cv2.putText(im2, "NCNN INT8 KL", (100,100), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 3)
im3 = cv2.putText(im3, "NCNN INT8 ACIQ", (100,100), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 3)
im = np.hstack([im1,im2,im3])

cv2.imwrite("cat.jpg",im)
