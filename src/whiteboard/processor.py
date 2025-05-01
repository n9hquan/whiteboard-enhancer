# whiteboard/processor.py

import cv2
import numpy as np

def load_image(path):
    """Load an image from file."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Cannot load image from {path}")
    return image

def detect_whiteboard(image):
    """
    Detect the largest contour that approximates a quadrilateral (the whiteboard).
    Returns the corner points ordered as (top-left, top-right, bottom-right, bottom-left)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by contour area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # If polygon has 4 corners, assume it's the whiteboard
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            # Order points
            return order_points(pts)

    raise ValueError("Whiteboard not detected.")

def order_points(pts):
    """Order corner points as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def correct_perspective(image, corners):
    """Apply homography to warp whiteboard to a top-down view."""
    (tl, tr, br, bl) = corners

    # Compute width and height of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for warp
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
