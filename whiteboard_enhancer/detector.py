"""
Board detection module for whiteboard enhancer.
"""

import cv2
import numpy as np


def detect_board(image, mode='auto'):
    """Detect board contour based on mode: whiteboard / smartboard / auto"""
    img_area = image.shape[0] * image.shape[1]

    def whiteboard_detector():
        """Original method: Canny + dilation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * img_area:
                return approx
        return None

    def smartboard_detector():
        """Smartboard method: Dark border detection using color threshold"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Mask dark colors (black/dark bezel detection)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 70])  # dark shades
        mask = cv2.inRange(hsv, lower, upper)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * img_area:
                return approx
        return None


    contour = whiteboard_detector()
    if contour is not None:
        print("Whiteboard detector succeeded.")
        return contour
    print("Whiteboard detector failed. Trying smartboard detector...")
    contour = smartboard_detector()
    if contour is not None:
        print("Smartboard detector succeeded.")
        return contour
    else:
        raise Exception("Error: Cannot detect whiteboard")
