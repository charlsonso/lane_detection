from cv2 import cv2
import numpy as np

def hough_redo(img):
    img = np.array(img, dtype=np.uint8)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    darken_img = gamma(grayscale_img, 0.5)
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_mask = isolate_color_mask(hls_img, np.array([0, 200, 0], dtype=np.uint8), np.array([200,255,255], dtype=np.uint8))
    color_mask_img = cv2.bitwise_and(darken_img, darken_img, mask=white_mask)
    gauss_img = cv2.GaussianBlur(color_mask_img, (5, 5), 0)
    edge_img = cv2.Canny(gauss_img, 70, 140)
    aoi_img = get_aoi(edge_img, lb=(.3, .7), rb=(.4, .7), lt=(.3, .4), rt = (.4, .4))
    lines = get_hough_lines(aoi_img) 
    new_img = draw_lines(img, lines)
    show(new_img)

def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2,y2), color, thickness)
    return img

def gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def isolate_color_mask(img, low, high):
    return cv2.inRange(img, low, high)

def show(img):
    cv2.imshow('hi', img)
    cv2.waitKey(0)

def get_aoi(img, lb=(0.1,1), rb=(0.95,1), lt=(0.4,0.6), rt=(0.6,0.6)):
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)
    
    left_bottom = [cols * lb[0], rows*lb[1]]
    right_bottom = [cols * rb[0], rows*rb[1]]
    left_top = [cols * lt[0], rows *lt[1]]
    right_top = [cols * rt[0], rows * rt[1]]
    
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])
    return cv2.bitwise_and(img, mask)
