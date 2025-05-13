import cv2
import numpy as np

def detect_board(image):
    """Detect whiteboard or digital board in an image"""
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    img_area = img_height * img_width
    
    # Define area thresholds
    min_area_ratio = 0.15  # Minimum contour area relative to image
    max_area_ratio = 0.95  # Maximum contour area relative to image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_gray, (15, 15), 0)
    
    # Try multiple edge detection methods and find the best contour
    def find_best_contour():
        candidates = []
        
        # Method 1: Edge detection with Canny
        for threshold1, threshold2 in [(30, 100), (50, 150), (100, 200)]:
            edges = cv2.Canny(blurred, threshold1, threshold2)
            # Dilate edges to connect broken lines
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours in the edge image
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
                    continue
                
                # Approximate the contour to find polygons
                peri = cv2.arcLength(contour, True)
                for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:  # Try multiple epsilon values
                    approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                    
                    # We're looking for quadrilaterals (4 points)
                    if len(approx) == 4:
                        # Check if it's convex
                        if cv2.isContourConvex(approx):
                            candidates.append(approx)
                            break
        
        # Method 2: Adaptive thresholding
        for block_size in [7, 11, 15]:
            for c in [2, 5, 8]:
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, block_size, c)
                
                # Morphological operations to clean up the binary image
                kernel = np.ones((3, 3), np.uint8)
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                # Find contours
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filter by area
                    area = cv2.contourArea(contour)
                    if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
                        continue
                    
                    # Approximate the contour
                    peri = cv2.arcLength(contour, True)
                    for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
                        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                        
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            candidates.append(approx)
                            break
        
        # Method 3: Color-based segmentation (for digital boards with specific colors)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different types of boards
        color_ranges = [
            # Dark backgrounds (black/dark boards)
            (np.array([0, 0, 0]), np.array([180, 255, 60])),
            # White backgrounds
            (np.array([0, 0, 200]), np.array([180, 30, 255]))
        ]
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
                    continue
                
                # Approximate the contour
                peri = cv2.arcLength(contour, True)
                for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                    
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        candidates.append(approx)
                        break
        
        # If no candidates found, return None with zero confidence
        if not candidates:
            return None, 0.0
        
        # Score each candidate and select the best one
        best_contour = None
        best_score = -1
        
        for contour in candidates:
            score = score_contour(contour)
            if score > best_score:
                best_score = score
                best_contour = contour
        
        # Calculate confidence based on the best score
        # Score range is 0-10, so normalize to 0-1
        confidence = min(best_score / 10.0, 1.0)
        
        return best_contour, confidence
    
    # Function to score a contour based on various criteria
    def score_contour(contour):
        # Calculate area score (prefer larger contours)
        area = cv2.contourArea(contour)
        area_ratio = area / img_area
        area_score = min(area_ratio * 20, 10)  # Scale to 0-10 range
        
        # Calculate aspect ratio score
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            return -1
            
        aspect_ratio = max(width, height) / min(width, height)
        # Prefer aspect ratios closer to standard whiteboard/paper ratios (1.3-1.8)
        aspect_score = 10 - min(abs(aspect_ratio - 1.5) * 2, 8)
        
        # Calculate position score (prefer centered contours)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center_offset_x = abs(cx - img_width/2) / (img_width/2)
            center_offset_y = abs(cy - img_height/2) / (img_height/2)
            position_score = 10 - 5 * (center_offset_x + center_offset_y) / 2
        else:
            position_score = 0
        
        # Calculate angle score (prefer rectangular shapes aligned with image axes)
        angle = abs(rect[2] % 90)
        angle = min(angle, 90 - angle)  # Normalize to 0-45 degrees
        angle_score = 10 - angle / 4.5  # Scale to 0-10 range
        
        # Calculate final score with weights
        final_score = (
            0.40 * area_score +      # Area is important
            0.25 * aspect_score +    # Aspect ratio is important
            0.20 * position_score +  # Position helps with centered whiteboards
            0.15 * angle_score       # Angle helps with aligned whiteboards
        )
        
        return final_score
    
    # Find the best contour and its confidence score
    best_contour, confidence = find_best_contour()
    
    # If no good contour found, use the whole image with a small margin and low confidence
    if best_contour is None:
        margin = int(min(img_width, img_height) * 0.02)  # 2% margin
        best_contour = np.array([
            [margin, margin],
            [img_width - margin, margin],
            [img_width - margin, img_height - margin],
            [margin, img_height - margin]
        ], dtype=np.int32)
        confidence = 0.1  # Very low confidence
    
    return best_contour, confidence