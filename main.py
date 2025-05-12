#!/usr/bin/env python3
"""
Whiteboard Enhancer - Main Application

A tool for scanning, enhancing, and converting whiteboard images to PDF.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import modules from the whiteboard_enhancer package
from whiteboard_enhancer.detector import detect_board
from whiteboard_enhancer.transformer import four_point_transform
from whiteboard_enhancer.enhancer import enhance_whiteboard
from whiteboard_enhancer.utils import save_as_pdf, display_image
from whiteboard_enhancer.ui import create_board_type_selector, display_contour, display_processing_steps


def process_image(image_path, mode='auto', display_steps=True):
    """
    Process a whiteboard image from start to finish
    
    Args:
        image_path: Path to the input image
        mode: Detection mode - 'auto', 'whiteboard', or 'smartboard'
        display_steps: Whether to display processing steps
        
    Returns:
        Dictionary containing processed images and file paths
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    filename = os.path.basename(image_path)
    
    # Display original image
    if display_steps:
        print("Original Image:")
        display_image(image)
    
    # Detect board contour
    contour = detect_board(image, mode)
    
    if contour is None:
        print("Could not detect whiteboard edges.")
        return {
            'original': image,
            'contour': None,
            'warped': None,
            'enhanced': None,
            'pdf_path': None,
            'enhanced_pdf_path': None
        }
    
    # Display detected contour
    if display_steps:
        display_contour(image, contour)
    
    # Warp perspective to get top-down view
    warped = four_point_transform(image, contour)
    
    # Display cropped whiteboard
    if display_steps:
        print("\nCropped Whiteboard:")
        display_image(warped)
    
    # Enhance the whiteboard
    enhanced = enhance_whiteboard(warped)
    
    # Display enhanced whiteboard
    if display_steps:
        print("\nEnhanced Whiteboard:")
        display_image(enhanced, cmap='gray')
    
    # Save PDFs
    pdf_path = save_as_pdf(warped, filename)
    enhanced_pdf_path = save_as_pdf(enhanced, "enhanced_" + filename)
    
    print("\nProcessing complete!")
    print(f"PDF files saved: {pdf_path} and {enhanced_pdf_path}")
    
    return {
        'original': image,
        'contour': contour,
        'warped': warped,
        'enhanced': enhanced,
        'pdf_path': pdf_path,
        'enhanced_pdf_path': enhanced_pdf_path
    }


def main():
    """Main function to run the whiteboard enhancer application"""
    try:
        # Try to import Google Colab specific modules
        from google.colab import files
        from IPython.display import display
        
        # We're in Google Colab
        print("Running in Google Colab environment")
        
        # Upload image
        uploaded = files.upload()
        
        if not uploaded:
            print("No file was uploaded.")
            return
        
        # Process the first uploaded file
        for filename in uploaded.keys():
            print(f'User uploaded file "{filename}"')
            image_path = filename
            break
        
        # Create and display board type selector
        mode_selector = create_board_type_selector()
        display(mode_selector)
        
        print("Please select the board type from dropdown above and then run the next cell to process.")
        
        # Note: In Colab, the user would need to run the next cell to continue processing
        # This is just a placeholder for the next cell's code
        mode = mode_selector.value
        
        # Process the image
        result = process_image(image_path, mode)
        
        # Display all processing steps
        display_processing_steps(
            result['original'], 
            result['contour'], 
            result['warped'], 
            result['enhanced'], 
            filename
        )
        
        # Download PDFs
        if result['pdf_path'] and result['enhanced_pdf_path']:
            files.download(result['pdf_path'])
            files.download(result['enhanced_pdf_path'])
    
    except ImportError:
        # We're not in Google Colab, use command line interface
        import argparse
        
        parser = argparse.ArgumentParser(description='Whiteboard Enhancer')
        parser.add_argument('image_path', help='Path to the whiteboard image')
        parser.add_argument('--mode', choices=['auto', 'whiteboard', 'smartboard'], 
                           default='auto', help='Board detection mode')
        parser.add_argument('--no-display', action='store_true', 
                           help='Disable display of processing steps')
        
        args = parser.parse_args()
        
        # Process the image
        process_image(args.image_path, args.mode, not args.no_display)


if __name__ == "__main__":
    main()
