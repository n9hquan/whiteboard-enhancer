"""
User interface components for whiteboard enhancer.
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np


def create_file_browser():
    """
    Create a file browser dialog to select an image file
    
    Returns:
        Path to the selected image file or None if canceled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Whiteboard Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )
    
    if file_path:
        return file_path
    return None


def display_contour(image, contour, title="Detected Whiteboard Contour"):
    """
    Display image with contour overlay
    
    Args:
        image: Input image
        contour: Detected contour
        title: Title for the plot
    """
    if contour is None:
        print("No contour to display")
        return
        
    debug_image = image.copy()
    cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()


def display_processing_steps(original, contour, warped, enhanced, filename=None):
    """
    Display all processing steps in a single figure
    
    Args:
        original: Original input image
        contour: Detected contour
        warped: Warped/transformed image
        enhanced: Enhanced image
        filename: Optional filename for the plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Whiteboard Enhancement Process: {filename}" if filename else "Whiteboard Enhancement Process")
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Contour detection
    if contour is not None:
        debug_image = original.copy()
        cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)
        axes[0, 1].imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 1].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 1].text(0.5, 0.5, "No contour detected", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[0, 1].transAxes, fontsize=12, color='red')
    axes[0, 1].set_title("Contour Detection")
    axes[0, 1].axis('off')
    
    # Warped image
    if warped is not None:
        axes[1, 0].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    else:
        axes[1, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[1, 0].text(0.5, 0.5, "Warping failed", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 0].transAxes, fontsize=12, color='red')
    axes[1, 0].set_title("Perspective Correction")
    axes[1, 0].axis('off')
    
    # Enhanced image
    if enhanced is not None:
        if len(enhanced.shape) == 2:  # Grayscale
            axes[1, 1].imshow(enhanced, cmap='gray')
        else:  # Color
            axes[1, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    else:
        axes[1, 1].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[1, 1].text(0.5, 0.5, "Enhancement failed", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=12, color='red')
    axes[1, 1].set_title("Enhanced Image")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
