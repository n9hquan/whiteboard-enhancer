"""
Utility functions for whiteboard enhancer.
"""

import cv2
import numpy as np
import img2pdf
import os


def upload_image():
    """
    Upload an image file through Colab interface
    
    Returns:
        Tuple of (image, filename) or (None, None) if no file was uploaded
    """
    try:
        from google.colab import files
        uploaded = files.upload()
        for filename in uploaded.keys():
            print(f'Uploaded {filename}')
            # Read the image with cv2.imdecode (flags=1 means color image)
            image = cv2.imdecode(np.frombuffer(uploaded[filename], np.uint8), 1)
            return image, filename
        return None, None
    except ImportError:
        print("This function is only available in Google Colab")
        return None, None


def save_as_pdf(image, filename):
    """
    Save image as PDF
    
    Args:
        image: Input image
        filename: Output filename
        
    Returns:
        Path to the saved PDF file
    """
    pdf_filename = filename.split('.')[0] + '.pdf'

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save temporary image
    temp_img = 'temp.jpg'
    cv2.imwrite(temp_img, image)

    # Convert to PDF
    with open(pdf_filename, "wb") as f:
        f.write(img2pdf.convert(temp_img))
    
    # Clean up temporary file
    if os.path.exists(temp_img):
        os.remove(temp_img)

    return pdf_filename


def display_image(image, title=None, cmap=None):
    """
    Display an image using matplotlib
    
    Args:
        image: Input image
        title: Optional title for the plot
        cmap: Optional colormap (use 'gray' for grayscale images)
    """
    import matplotlib.pyplot as plt
    
    # Convert BGR to RGB if needed (color image and no specific colormap)
    if cmap is None and len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
