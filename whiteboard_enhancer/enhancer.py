"""
Image enhancement module for whiteboard enhancer.
"""

import cv2


def enhance_whiteboard(image):
    """
    Enhance whiteboard: Contrast, binarization, and noise reduction
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Enhanced grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Adaptive threshold (strong binarization)
    enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 9)

    # Step 3: Denoising (optional but improves clarity)
    enhanced = cv2.medianBlur(enhanced, 3)

    return enhanced
