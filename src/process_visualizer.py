import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the current directory to the path to ensure imports work
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the necessary functions from the project
try:
    from detector import detect_board
    from transformer import four_point_transform
    from enhancer import enhance_whiteboard
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project directory")
    sys.exit(1)

def visualize_processing_steps(image_path, output_dir=None, detailed_detection=False):
    """
    Visualize each step of the whiteboard enhancement process
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images (optional)
        detailed_detection: Whether to show detailed whiteboard detection steps
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    print(f"Loading image: {image_path}")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Create a figure with subplots for visualization
    if detailed_detection:
        # Create two figures for detailed detection
        fig1, axs1 = plt.subplots(2, 3, figsize=(15, 10))
        fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
        # Keep the original 2D structure for main visualization
        # and create a flattened array for detailed steps
        axs_detailed = np.concatenate([axs1.flatten(), axs2.flatten()])
    else:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    plt.tight_layout(pad=3.0)
    
    # Step 1: Display the original image
    print("Step 1: Original Image")
    if detailed_detection:
        axs1[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs1[0, 0].set_title('1. Original Image')
        axs1[0, 0].axis('off')
    else:
        axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('1. Original Image')
        axs[0, 0].axis('off')
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '1_original.jpg'), original_image)
    
    # Step 2: Detect whiteboard
    print("Step 2: Detecting Whiteboard")
    
    # If detailed detection is requested, show the intermediate steps
    if detailed_detection:
        print("  Showing detailed detection steps...")
        # Create a copy of the original image for visualization
        detection_vis = original_image.copy()
        
        # Step 2.1: Convert to grayscale for detection
        gray_detect = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        axs2[0, 0].imshow(gray_detect, cmap='gray')
        axs2[0, 0].set_title('2.1. Grayscale for Detection')
        axs2[0, 0].axis('off')
        
        # Step 2.2: Apply CLAHE to enhance contrast for detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_detect)
        axs2[0, 1].imshow(enhanced_gray, cmap='gray')
        axs2[0, 1].set_title('2.2. CLAHE Enhanced')
        axs2[0, 1].axis('off')
        
        # Step 2.3: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced_gray, (15, 15), 0)
        axs2[0, 2].imshow(blurred, cmap='gray')
        axs2[0, 2].set_title('2.3. Gaussian Blur')
        axs2[0, 2].axis('off')
        
        # Step 2.4: Edge detection with Canny
        edges = cv2.Canny(blurred, 50, 150)
        axs2[1, 0].imshow(edges, cmap='gray')
        axs2[1, 0].set_title('2.4. Canny Edge Detection')
        axs2[1, 0].axis('off')
        
        # Step 2.5: Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        axs2[1, 1].imshow(dilated, cmap='gray')
        axs2[1, 1].set_title('2.5. Dilated Edges')
        axs2[1, 1].axis('off')
        
        # Step 2.6: Find contours in the edge image
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(original_image)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        axs2[1, 2].imshow(contour_img)
        axs2[1, 2].set_title(f'2.6. Found Contours ({len(contours)} contours)')
        axs2[1, 2].axis('off')
        
        # Save the detailed detection steps
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, '2.1_gray_detect.jpg'), gray_detect)
            cv2.imwrite(os.path.join(output_dir, '2.2_enhanced_gray.jpg'), enhanced_gray)
            cv2.imwrite(os.path.join(output_dir, '2.3_blurred.jpg'), blurred)
            cv2.imwrite(os.path.join(output_dir, '2.4_edges.jpg'), edges)
            cv2.imwrite(os.path.join(output_dir, '2.5_dilated.jpg'), dilated)
            cv2.imwrite(os.path.join(output_dir, '2.6_contours.jpg'), contour_img)
        
        # Final contour selection
        final_contour_img = original_image.copy()
    
    # Run the actual detection
    contour, confidence = detect_board(original_image)
    
    # Create a copy of the original image for visualization
    detection_vis = original_image.copy()
    cv2.drawContours(detection_vis, [contour], -1, (0, 255, 0), 3)
    
    # Display detection result
    if detailed_detection:
        # Update the final contour image in the detailed view
        cv2.drawContours(final_contour_img, [contour], -1, (0, 255, 0), 3)
        
        # Also show in the main figure
        axs1[0, 1].imshow(cv2.cvtColor(detection_vis, cv2.COLOR_BGR2RGB))
        axs1[0, 1].set_title(f'2. Whiteboard Detection\nConfidence: {confidence:.2f}')
        axs1[0, 1].axis('off')
    else:
        # Standard display
        axs[0, 1].imshow(cv2.cvtColor(detection_vis, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title(f'2. Whiteboard Detection\nConfidence: {confidence:.2f}')
        axs[0, 1].axis('off')
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '2_detection.jpg'), detection_vis)
        if detailed_detection:
            cv2.imwrite(os.path.join(output_dir, '2.10_final_contour.jpg'), final_contour_img)
    
    # Step 3: Apply perspective transformation
    print("Step 3: Applying Perspective Transformation")
    warped = four_point_transform(original_image, contour)
    
    # Display warped image
    if detailed_detection:
        axs1[0, 2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axs1[0, 2].set_title('3. Perspective Transformation')
        axs1[0, 2].axis('off')
    else:
        axs[0, 2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axs[0, 2].set_title('3. Perspective Transformation')
        axs[0, 2].axis('off')
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '3_warped.jpg'), warped)
    
    # Step 4: Convert to grayscale (intermediate step)
    print("Step 4: Converting to Grayscale")
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Display grayscale image
    if detailed_detection:
        axs1[1, 0].imshow(gray, cmap='gray')
        axs1[1, 0].set_title('4. Grayscale Conversion')
        axs1[1, 0].axis('off')
    else:
        axs[1, 0].imshow(gray, cmap='gray')
        axs[1, 0].set_title('4. Grayscale Conversion')
        axs[1, 0].axis('off')
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '4_grayscale.jpg'), gray)
    
    # Step 5: Apply CLAHE enhancement (intermediate step)
    print("Step 5: Applying CLAHE Enhancement")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gray)
    
    # Display CLAHE enhanced image
    if detailed_detection:
        axs1[1, 1].imshow(clahe_enhanced, cmap='gray')
        axs1[1, 1].set_title('5. CLAHE Enhancement')
        axs1[1, 1].axis('off')
    else:
        axs[1, 1].imshow(clahe_enhanced, cmap='gray')
        axs[1, 1].set_title('5. CLAHE Enhancement')
        axs[1, 1].axis('off')
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '5_clahe.jpg'), clahe_enhanced)
    
    # Step 6: Final enhanced result
    print("Step 6: Final Enhancement")
    enhanced = enhance_whiteboard(warped)
    
    # Display final enhanced image
    if detailed_detection:
        axs1[1, 2].imshow(enhanced, cmap='gray')
        axs1[1, 2].set_title('6. Final Enhanced Result')
        axs1[1, 2].axis('off')
    else:
        axs[1, 2].imshow(enhanced, cmap='gray')
        axs[1, 2].set_title('6. Final Enhanced Result')
        axs[1, 2].axis('off')
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '6_enhanced.jpg'), enhanced)
    
    # Adjust layout and display
    if detailed_detection:
        # Set titles for the figures
        fig1.suptitle('Main Whiteboard Enhancement Steps', fontsize=16)
        fig2.suptitle('Detailed Whiteboard Detection Steps', fontsize=16)
        
        # Adjust layout
        fig1.tight_layout()
        fig2.tight_layout()
        fig1.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
        fig2.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
        
        # Save the complete visualization
        if output_dir:
            fig1.savefig(os.path.join(output_dir, 'main_steps_visualization.png'), dpi=300, bbox_inches='tight')
            fig2.savefig(os.path.join(output_dir, 'detailed_detection_visualization.png'), dpi=300, bbox_inches='tight')
        
        # Show the visualizations
        plt.figure(fig1.number)
        plt.show()
        plt.figure(fig2.number)
        plt.show()
    else:
        # Standard mode
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save the complete visualization
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'complete_visualization.png'), dpi=300, bbox_inches='tight')
        
        # Show the visualization
        plt.show()
    
    print("Processing complete!")
    if output_dir:
        print(f"Output images saved to: {output_dir}")
    
    return {
        'original': original_image,
        'detection': detection_vis,
        'warped': warped,
        'grayscale': gray,
        'clahe': clahe_enhanced,
        'enhanced': enhanced
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize whiteboard enhancement process steps')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--output', '-o', help='Directory to save output images')
    parser.add_argument('--detailed-detection', '-d', action='store_true', 
                        help='Show detailed whiteboard detection steps')
    
    args = parser.parse_args()
    
    visualize_processing_steps(args.image_path, args.output, args.detailed_detection)

if __name__ == "__main__":
    main()
