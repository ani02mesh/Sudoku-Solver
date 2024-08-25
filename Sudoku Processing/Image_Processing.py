import cv2

def Processing(image):
    # make image Gray scale
    img_gray = cv2. cvtColor(image, cv2. COLOR_BGR2GRAY)
    # make image blur
    blurr = cv2.GaussianBlur(img_gray, (7, 7), 0)
    imgThres = cv2.adaptiveThreshold(blurr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,11,2)
    imgThres = 255 - imgThres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # morph so as to remove noice from the image
    morph = cv2.morphologyEx(imgThres, cv2.MORPH_OPEN, kernel)
    final = cv2.dilate(morph,kernel)
    return final
