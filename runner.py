from cv2 import cv2
import os
import hough
import numpy as np
import hough_redo


image_folder = 'clear_lanes' 
hough_folder = 'ht_clear_lanes'
def ht_road():
    list_of_images = [os.path.join(image_folder, i) for i in os.listdir(image_folder)] 
    for i in list_of_images:
        print(i)
        img = cv2.imread(i)
        print(img)
        img = np.array(img, dtype=np.uint8)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        new_img = hough.hough_transform(img, edges) 
        new_path = os.path.join(hough_folder, 'h'+os.path.basename(i))
        cv2.imwrite(new_path, new_img)

clear_lanes = "clear_lanes/1-IKPBOOKUeKqvnyQWLJ6U9A.jpeg"
dataset_img = "images/01410.jpg"

def ht_redo():
    img = cv2.imread(dataset_img)
    hough_redo.hough_redo(img)

ht_redo()
