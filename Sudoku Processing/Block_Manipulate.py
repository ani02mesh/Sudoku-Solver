import cv2
import numpy as np

def warp(corner,Image):
    corners = np.array(corner, dtype='float32')
    top_left, top_right, bot_right, bot_left = corners
    width = int(max([np.linalg.norm(top_right - top_left),
                    np.linalg.norm(bot_right - bot_left),
                    np.linalg.norm(top_right - bot_right),
                    np.linalg.norm(top_left - bot_left)
                    ]))
    output = np.array([[0,0],[width-1,0],[width-1,width-1],[0,width-1]],dtype='float32')
    M = cv2.getPerspectiveTransform(corners,output)
    out = cv2.warpPerspective(Image,M,(width, width))
    return out


def grid_lines(img,vertex):
    replica = img.copy()
    kernel_size = replica.shape[vertex]//10
    if vertex == 0:
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, kernel_size))
    else:
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(kernel_size, 1))
    replica = cv2.erode(replica, kernel)
    clone = cv2.dilate(replica, kernel)
    return clone

def row_col_line(img):
    horizontal = grid_lines(img,1)
    vertical = grid_lines(img,0)
    return horizontal,vertical


def create_mask(hor,ver):
    grid = cv2.add(hor,ver)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)
    clone = grid.copy()
    lines = np.squeeze(pts)
    for rho,theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(clone, pt1, pt2, (255, 255, 255), 4)
    mask = cv2.bitwise_not(clone)
    return mask


def individual_square(Image):
    squares = []
    width = Image.shape[0] // 9
    
    for j in range(9):
        for i in range(9):
            pt1 = (i*width+8,j*width+8)
            pt2 = ((i+1)*width-4,(j+1)*width-3)
            squares.append(Image[pt1[1]:pt2[1], pt1[0]:pt2[0]])
    return squares

