import cv2
import numpy as np

def draw_digits_on_warped(warped_img, solved_puzzle, squares_processed):
    width = warped_img.shape[0] // 9 
    solved_puzzle = solved_puzzle.flatten()
    squares_processed = squares_processed.flatten()
    
    index = 0
    for j in range(9):
        for i in range(9):
            if squares_processed[index] == 0:
                p1 = (i * width, j * width)  # Top left corner of a bounding box
                p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1


def unwarp_image(img_src, img_dest, pts):
    pts = np.array(pts)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts, 0, 16)

    dst_img = cv2.add(img_dest, warped)


    return dst_img

