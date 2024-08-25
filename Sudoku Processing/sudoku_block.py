import cv2
import numpy as np
import operator

def find_coordinate(poly,min_max,add_sub):
    mno = min_max(enumerate([add_sub(pt[0][0],pt[0][1]) for pt in poly]),key=operator.itemgetter(1))[0]
    return poly[mno][0][0],poly[mno][0][1]


def find_block(img,Image):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt)>1000:
            polygon = cnt
            break
    if polygon is not None:
        top_left = find_coordinate(polygon,min,np.add)
        top_right = find_coordinate(polygon,max,np.subtract)
        bot_left = find_coordinate(polygon,min,np.subtract)
        bot_right = find_coordinate(polygon,max,np.add)
        
        if bot_right[1] - top_right[1] == 0:
            return []
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []
        
        # cv2.drawContours(Image,[polygon],0,(0,255,0),3)
        # [cv2.circle(Image, x, 7, (255, 0, 0), cv2.FILLED) for x in [top_left, top_right, bot_right, bot_left]]
        return [top_left, top_right, bot_right, bot_left]
    return []

