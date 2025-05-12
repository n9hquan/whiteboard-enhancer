"""
Image enhancement module for whiteboard enhancer.
"""

import cv2


def enhance_whiteboard(image, params=None):
    """
    Enhance whiteboard: Denoising, contrast enhancement, and thresholding
    
    Args:
        image: Input image (BGR format)
        params: Dictionary of enhancement parameters (optional)
            - denoise_h: Denoising strength (default: 10)
            - clahe_clip: CLAHE clip limit (default: 2.0)
            - clahe_grid: CLAHE grid size (default: 8)
            - adaptive_block: Adaptive threshold block size (default: 15)
            - adaptive_c: Adaptive threshold C value (default: 9)
            - use_adaptive: Whether to use adaptive thresholding (default: True)
            - threshold: Global threshold value (default: 160)
            - blur_size: Final blur kernel size (default: 3)
        
    Returns:
        Enhanced grayscale image
    """
    # Default parameters
    default_params = {
        'denoise_h': 10,
        'clahe_clip': 2.0,
        'clahe_grid': 8,
        'adaptive_block': 15,
        'adaptive_c': 9,
        'use_adaptive': True,
        'threshold': 160,
        'blur_size': 3
    }
    
    # Use provided parameters or defaults
    if params is None:
        params = default_params
    else:
        # Fill in any missing parameters with defaults
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
    
    # Step 0: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Initial denoising to remove noise before processing
    denoised = cv2.fastNlMeansDenoising(gray, None, h=params['denoise_h'])
    
    # Step 2: Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], 
                           tileGridSize=(params['clahe_grid'], params['clahe_grid']))
    enhanced = clahe.apply(denoised)
    
    # Step 3: Thresholding (either adaptive or global based on parameters)
    if params['use_adaptive']:
        # Adaptive thresholding for varying lighting conditions
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, params['adaptive_block'], params['adaptive_c'])
    else:
        # Global thresholding for more uniform results
        _, binary = cv2.threshold(enhanced, params['threshold'], 255, cv2.THRESH_BINARY)
    
    # Step 4: Final cleanup with median blur to remove salt-and-pepper noise
    if params['blur_size'] > 0:
        final = cv2.medianBlur(binary, params['blur_size'])
    else:
        final = binary
    
    return final


def enhance_whiteboard_color(image, params=None):
    """
    Enhance whiteboard while preserving color information
    
    Args:
        image: Input image (BGR format)
        params: Dictionary of enhancement parameters (same as enhance_whiteboard)
        
    Returns:
        Enhanced color image
    """
    # Default parameters
    default_params = {
        'denoise_h': 10,
        'clahe_clip': 2.0,
        'clahe_grid': 8,
        'adaptive_block': 15,
        'adaptive_c': 9,
        'use_adaptive': True,
        'threshold': 160,
        'blur_size': 3,
        'saturation_boost': 1.2  # Boost colors slightly
    }
    
    # Use provided parameters or defaults
    if params is None:
        params = default_params
    else:
        # Fill in any missing parameters with defaults
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
    
    # Step 1: Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Step 2: Enhance the value (brightness) channel
    enhanced_v = enhance_whiteboard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), params)
    
    # Step 3: Boost saturation slightly to make colors more vivid
    s = cv2.multiply(s, params['saturation_boost'])
    s = cv2.min(s, 255)  # Ensure we don't exceed 255
    
    # Step 4: Merge channels back together
    enhanced_hsv = cv2.merge([h, s, enhanced_v])
    
    # Step 5: Convert back to BGR
    enhanced_color = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    return enhanced_color
