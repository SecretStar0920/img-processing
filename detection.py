import cv2
import numpy as np
import os
from os import listdir

def captch_ex(file_name):    
    file_path = "dataset/" + file_name

    img_origin = cv2.imread(file_path)
    img_denoised = cv2.fastNlMeansDenoising(img_origin)

    gray = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
 

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilated = cv2.dilate(thresh1, rect_kernel, iterations=2)  # dilate , more the iteration more the dilation

    # cv2.imshow('dilated', dilated)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 300 and h < 500:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img_origin, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # write original image with added contours to disk
    # cv2.imshow('captcha_result', img)    
    cv2.imwrite("instructions/" + file_name, img_origin)
    cv2.waitKey()
# captch_ex("dataset/00.png")
image_dir = "./dataset"
for images in os.listdir(image_dir):
    if (images.endswith(".png")):
        captch_ex(images)