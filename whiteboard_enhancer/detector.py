"""
Board detection module for whiteboard enhancer.
"""

import cv2
import numpy as np


def detect_board(image, mode='auto'):
    """Detect board contour based on mode: whiteboard / smartboard / auto"""
    img_area = image.shape[0] * image.shape[1]

    def whiteboard_detector():
        """Improved method: More robust edge detection with noise reduction"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(bilateral)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        # Use adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours in the processed image
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        valid_contours = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(approx)
            
            # Check if it's a quadrilateral with reasonable area
            if len(approx) == 4 and area > 0.05 * img_area:
                # Additional check for aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for whiteboards
                    valid_contours.append(approx)
        
        # Return the largest valid contour
        if valid_contours:
            return valid_contours[0]
            
        # If no valid contour found with the first method, try Canny edge detection
        edges = cv2.Canny(blurred, 30, 200)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * img_area:
                return approx
                
        return None

    def smartboard_detector():
        """Improved smartboard method: Better dark border detection with noise filtering"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create multiple masks for different dark colors
        # Black/very dark colors
        lower1 = np.array([0, 0, 0])
        upper1 = np.array([180, 255, 50])
        
        # Dark gray colors
        lower2 = np.array([0, 0, 51])
        upper2 = np.array([180, 30, 80])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply noise reduction
        kernel_small = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(mask_cleaned, kernel, iterations=2)
        
        # Close gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area, shape, and aspect ratio
        valid_contours = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(approx)
            
            # Check if it's a quadrilateral with reasonable area
            if len(approx) == 4 and area > 0.05 * img_area:
                # Additional check for aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for whiteboards
                    valid_contours.append(approx)
        
        # Return the largest valid contour
        if valid_contours:
            return valid_contours[0]
        
        return None
    def hough_lines_detector():
        """Fallback method using Hough lines transformation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(bilateral, (5, 5), 0)
        
        # Apply Canny edge detection with lower thresholds
        edges = cv2.Canny(blurred, 20, 100)
        
        # Apply dilation to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            return None
            
        # Create a blank image to draw lines on
        line_image = np.zeros_like(gray)
        
        # Draw lines on the blank image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 5)
            
        # Dilate the line image to connect nearby lines
        line_image = cv2.dilate(line_image, kernel, iterations=2)
        
        # Find contours in the line image
        contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * img_area:
                # Check aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:
                    return approx
                    
        return None

    # Try each detection method in sequence
    contour = whiteboard_detector()
    if contour is not None:
        print("Whiteboard detector succeeded.")
        return contour
        
    print("Whiteboard detector failed. Trying smartboard detector...")
    contour = smartboard_detector()
    if contour is not None:
        print("Smartboard detector succeeded.")
        return contour
        
    print("Smartboard detector failed. Trying Hough lines detector...")
    contour = hough_lines_detector()
    if contour is not None:
        print("Hough lines detector succeeded.")
        return contour
        
    # If all detection methods fail, try to use the entire image as a fallback
    print("All detection methods failed. Using entire image as fallback.")
    h, w = image.shape[:2]
    fallback_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32).reshape((-1, 1, 2))
    return fallback_contour
