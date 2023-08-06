import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import rotate
import imutils
from skimage.filters import threshold_local

np.seterr(divide='ignore', invalid='ignore')
# read image as grayscale
def skewcorrection (img):
    # convert to binary

    
   
    bin_img = 1 - (img.reshape((img.shape[0], img.shape[1])) / 255.0)
    
    def find_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score
    delta = 1
    limit = 10
    angles = np.arange(-limit, limit+delta, delta)
    scores = [] 
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
   
    image1 = cv2.imread(img_path)
    height, width =image1.shape[:2]
# get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)

# using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle = best_angle, scale=1)

# rotate the image using cv2.warpAffine
    result = cv2.warpAffine(src=image1, M=rotate_matrix, dsize=(width, height))
    return result

def equalize(img):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9))
    close = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(img)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    cv2.imwrite("preprocess.jpg", res)

def removebackground (img):
# threshold
    # img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)[1]
    

# apply morphology
    kernel = np.ones((15,15), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((20,20), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)
    

# get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
        big_contour = c


# get bounding box
    x,y,w,h = cv2.boundingRect(big_contour)

# draw filled contour on black background
    mask = np.zeros_like(gray)
    mask = cv2.merge([mask,mask,mask])
    cv2.drawContours(mask, [big_contour], -1, (255,255,255), cv2.FILLED)

# apply mask to input
    result1 = img.copy()
    result1 = cv2.bitwise_and(result1, mask)

# crop result
    result2 = result1[y:y+h, x:x+w]
    cv2.imwrite("preprocess.jpg", result2)
    path = "preprocess.jpg"
    return path

    
    


def preprocess(img_path):
    img = skewcorrection(img_path)
    img_2 = removebackground(img)
    # equalize(img)
    
    path = "preprocess.jpg"
    # gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite(path, gray_img)
    return path

# img = skewcorrection(r'D:\Thesis\LVTN_VanThanhThuan\Dataset\Dataset_test2\test (43).jpg')
# # a = cv2.imread(img)
# plt.imshow(img)
# plt.show()
