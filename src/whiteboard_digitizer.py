import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import argparse


class WhiteboardDigitizer:
    """
    A class that implements the Whiteboard-It functionality as described in the paper
    by Zhengyou Zhang and Li-wei He from Microsoft Research.
    """

    def __init__(self):
        """Initialize the whiteboard digitizer with default parameters."""
        self.edge_threshold = 40
        self.hough_threshold = 0.05  # 5% of max votes
        self.cell_size = 15  # for white balancing

    def process_image(self, image_path, manual_corners=None):
        """
        Process a whiteboard image to produce a clean, rectified document.
        
        Args:
            image_path: Path to the image file
            manual_corners: Optional list of 4 corner points if auto-detection fails
        
        Returns:
            The processed whiteboard image
        """
        # Read the image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to RGB (OpenCV uses BGR by default)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Make a copy for displaying results
        self.display_image = self.image.copy()
        
        # Step 1: Detect whiteboard corners
        if manual_corners is not None:
            self.corners = np.array(manual_corners, dtype=np.float32)
        else:
            self.detect_whiteboard()
        
        # Step 2: Estimate aspect ratio
        aspect_ratio = self.estimate_aspect_ratio()
        print(f"Estimated aspect ratio: {aspect_ratio:.3f}")
        
        # Step 3: Rectify the image
        self.rectified_image = self.rectify_image(aspect_ratio)
        
        # Step 4: White balance the image
        self.enhanced_image = self.enhance_colors()
        
        return self.enhanced_image
    
    def detect_whiteboard(self):
        """Detect the whiteboard boundaries in the image."""
        print("Detecting whiteboard boundaries...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection using Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and orientation
        magnitude = np.abs(sobel_x) + np.abs(sobel_y)
        edges = (magnitude > self.edge_threshold).astype(np.uint8) * 255
        
        # Calculate orientation for each edge pixel
        orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        
        # Hough transform to detect lines
        lines = self.hough_transform(edges, orientation)
        
        # Form and verify quadrangles
        self.corners = self.find_best_quadrangle(lines, edges, orientation)
        
        # Display corners
        for point in self.corners:
            cv2.circle(self.display_image, tuple(point.astype(int)), 5, (255, 0, 0), -1)
        
        # Connect corners to show the detected quadrangle
        for i in range(4):
            pt1 = tuple(self.corners[i].astype(int))
            pt2 = tuple(self.corners[(i + 1) % 4].astype(int))
            cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)
    
    def hough_transform(self, edges, orientation):
        """
        Perform Hough transform to detect lines in the edge image.
        
        Args:
            edges: Binary edge image
            orientation: Orientation of each edge pixel
        
        Returns:
            List of lines in (rho, theta) format
        """
        height, width = edges.shape
        diagonal = np.sqrt(height**2 + width**2)
        
        # Parameters for Hough space
        rho_resolution = 5
        theta_resolution = 2
        rho_range = int(2 * diagonal / rho_resolution)
        theta_range = 180  # -180 to 180 with 2 degree steps
        
        # Initialize accumulator
        accumulator = np.zeros((rho_range, theta_range), dtype=np.int32)
        
        # Build accumulator
        y_indices, x_indices = np.where(edges > 0)
        for y, x in zip(y_indices, x_indices):
            theta = orientation[y, x]
            # Convert to 0-179 range
            theta_idx = int((theta + 180) / 2) % 180
            rho = x * np.cos(np.radians(theta)) + y * np.sin(np.radians(theta))
            rho_idx = int((rho + diagonal) / rho_resolution)
            if 0 <= rho_idx < rho_range and 0 <= theta_idx < theta_range:
                accumulator[rho_idx, theta_idx] += 1
        
        # Find peaks in accumulator
        max_votes = np.max(accumulator)
        threshold = max_votes * self.hough_threshold
        lines = []
        
        for rho_idx in range(rho_range):
            for theta_idx in range(theta_range):
                if accumulator[rho_idx, theta_idx] > threshold:
                    # Check if it's a local maximum
                    is_max = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = rho_idx + dy, theta_idx + dx
                            if (0 <= ny < rho_range and 0 <= nx < theta_range and 
                                accumulator[ny, nx] > accumulator[rho_idx, theta_idx]):
                                is_max = False
                                break
                        if not is_max:
                            break
                    
                    if is_max:
                        rho = (rho_idx * rho_resolution) - diagonal
                        theta = (theta_idx * 2) - 180
                        lines.append((rho, theta, accumulator[rho_idx, theta_idx]))
        
        # Sort lines by vote count (descending)
        lines.sort(key=lambda x: x[2], reverse=True)
        return lines
    
    def find_best_quadrangle(self, lines, edges, orientation):
        """
        Form and verify quadrangles from detected lines.
        
        Args:
            lines: List of lines in (rho, theta) format
            edges: Binary edge image
            orientation: Orientation of each edge pixel
        
        Returns:
            Array of corners of the best quadrangle
        """
        height, width = edges.shape
        best_quality = 0
        best_corners = None
        
        # Try different combinations of 4 lines
        for i in range(min(len(lines), 20)):
            for j in range(i + 1, min(len(lines), 20)):
                for k in range(j + 1, min(len(lines), 20)):
                    for l in range(k + 1, min(len(lines), 20)):
                        line_set = [lines[i], lines[j], lines[k], lines[l]]
                        
                        # Check if lines form a valid quadrangle
                        if not self.is_valid_quadrangle(line_set, width, height):
                            continue
                        
                        # Compute intersection points
                        corners = self.compute_corners(line_set)
                        if corners is None:
                            continue
                        
                        # Verify with edge support
                        quality = self.verify_quadrangle(corners, edges, orientation)
                        if quality > best_quality:
                            best_quality = quality
                            best_corners = corners
        
        if best_corners is None:
            # Fallback to using the image boundaries if no quadrangle is found
            best_corners = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
        
        # Order corners in clockwise order: top-left, top-right, bottom-right, bottom-left
        center = np.mean(best_corners, axis=0)
        angles = np.arctan2(best_corners[:, 1] - center[1], best_corners[:, 0] - center[0])
        sorted_indices = np.argsort(-angles)  # Sort clockwise
        top_left_idx = np.argmin(np.sum(best_corners, axis=1))
        rotation = np.where(sorted_indices == top_left_idx)[0][0]
        sorted_indices = np.roll(sorted_indices, -rotation)
        
        return best_corners[sorted_indices]
    
    def is_valid_quadrangle(self, line_set, width, height):
        """Check if the lines can form a valid quadrangle."""
        rhos = [line[0] for line in line_set]
        thetas = [np.radians(line[1]) for line in line_set]
        
        # Check for parallel lines (opposite orientation)
        for i in range(4):
            for j in range(i + 1, 4):
                if abs(abs(thetas[i] - thetas[j]) - np.pi) < np.radians(30):
                    # Lines are roughly parallel, check if they're far enough apart
                    rho_diff = abs(rhos[i] - rhos[j])
                    min_distance = min(width, height) / 5
                    if rho_diff < min_distance:
                        return False
        
        # Check for perpendicular lines
        perpendicular_count = 0
        for i in range(4):
            for j in range(i + 1, 4):
                angle_diff = abs(abs(thetas[i] - thetas[j]) - np.pi/2)
                if angle_diff < np.radians(30):
                    perpendicular_count += 1
        
        return perpendicular_count >= 2
    
    def compute_corners(self, line_set):
        """Compute the intersection points of the lines."""
        corners = []
        
        for i in range(4):
            for j in range(i + 1, 4):
                # Compute intersection of lines i and j
                rho1, theta1, _ = line_set[i]
                rho2, theta2, _ = line_set[j]
                
                theta1_rad = np.radians(theta1)
                theta2_rad = np.radians(theta2)
                
                # Check if lines are nearly parallel
                if abs(np.sin(theta1_rad - theta2_rad)) < 1e-10:
                    continue
                
                # Solve for intersection
                A = np.array([
                    [np.cos(theta1_rad), np.sin(theta1_rad)],
                    [np.cos(theta2_rad), np.sin(theta2_rad)]
                ])
                b = np.array([rho1, rho2])
                try:
                    point = np.linalg.solve(A, b)
                    corners.append(point)
                except np.linalg.LinAlgError:
                    continue
        
        # If we have exactly 4 corners, return them
        if len(corners) == 4:
            return np.array(corners)
        return None
    
    def verify_quadrangle(self, corners, edges, orientation):
        """
        Verify if the quadrangle has good edge support.
        
        Args:
            corners: Corner points of the quadrangle
            edges: Binary edge image
            orientation: Orientation of each edge pixel
        
        Returns:
            Quality measure of the quadrangle (ratio of supporting edges to perimeter)
        """
        height, width = edges.shape
        perimeter = 0
        edge_count = 0
        
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i + 1) % 4]
            
            # Skip if any point is outside the image
            if (pt1[0] < 0 or pt1[0] >= width or pt1[1] < 0 or pt1[1] >= height or
                pt2[0] < 0 or pt2[0] >= width or pt2[1] < 0 or pt2[1] >= height):
                return 0
            
            # Calculate line parameters
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = np.sqrt(dx**2 + dy**2)
            perimeter += length
            
            # Skip if line is too short
            if length < 10:
                return 0
            
            # Sample points along the line and count edges
            num_samples = int(length)
            for t in np.linspace(0, 1, num_samples):
                x = int(pt1[0] + t * dx)
                y = int(pt1[1] + t * dy)
                
                # Check a small neighborhood for edge support
                for ny in range(max(0, y-3), min(height, y+4)):
                    for nx in range(max(0, x-3), min(width, x+4)):
                        if edges[ny, nx] > 0:
                            # Check if the edge orientation is compatible with the line
                            line_orientation = np.degrees(np.arctan2(dy, dx)) % 180
                            edge_orientation = orientation[ny, nx] % 180
                            if abs(line_orientation - edge_orientation) < 30:
                                edge_count += 1
                                break
                    else:
                        continue
                    break
        
        # Calculate quality as the ratio of supporting edges to perimeter
        return edge_count / perimeter if perimeter > 0 else 0
    
    def estimate_aspect_ratio(self):
        """
        Estimate the aspect ratio of the whiteboard based on the perspective projection.
        
        Returns:
            Estimated aspect ratio (width/height)
        """
        print("Estimating aspect ratio...")
        
        # Assume the principal point is at the image center
        height, width = self.image.shape[:2]
        u0 = width / 2
        v0 = height / 2
        s = 1  # Square pixels
        
        # Convert corners to homogeneous coordinates
        m1 = np.array([self.corners[0][0], self.corners[0][1], 1])
        m2 = np.array([self.corners[1][0], self.corners[1][1], 1])
        m3 = np.array([self.corners[3][0], self.corners[3][1], 1])
        m4 = np.array([self.corners[2][0], self.corners[2][1], 1])
        
        # Calculate k2 and k3 (equation 11 and 12 in the paper)
        k2_num = np.dot(np.cross(m1, m4), m3)
        k2_den = np.dot(np.cross(m2, m4), m3)
        k3_num = np.dot(np.cross(m1, m4), m2)
        k3_den = np.dot(np.cross(m3, m4), m2)
        
        if abs(k2_den) < 1e-10 or abs(k3_den) < 1e-10:
            print("Warning: Unable to estimate aspect ratio accurately. Using fallback.")
            return 4/3  # Default aspect ratio as fallback
        
        k2 = k2_num / k2_den
        k3 = k3_num / k3_den
        
        # Calculate n2 and n3 (equation 14 and 16)
        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1
        
        # Estimate focal length (equation 21)
        n21, n22, n23 = n2
        n31, n32, n33 = n3
        
        if abs(n23 * n33) < 1e-10:
            print("Warning: Unable to estimate focal length accurately. Using fallback.")
            f = max(width, height)  # Fallback focal length
        else:
            num = -(1/n23/n33) * s**2 * (
                (n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0**2) * s**2 +
                (n22*n32 - (n22*n33 + n23*n32)*v0 + n23*n33*v0**2)
            )
            if num < 0:
                print("Warning: Negative focal length squared. Using fallback.")
                f = max(width, height)  # Fallback focal length
            else:
                f = np.sqrt(num)
        
        # Construct the camera intrinsic matrix
        A = np.array([
            [f, 0, u0],
            [0, s*f, v0],
            [0, 0, 1]
        ])
        
        # Calculate the aspect ratio (equation 20)
        A_inv = np.linalg.inv(A)
        A_inv_t = A_inv.T
        
        num = np.dot(n2, np.dot(A_inv_t, np.dot(A_inv, n2)))
        den = np.dot(n3, np.dot(A_inv_t, np.dot(A_inv, n3)))
        
        if den < 1e-10:
            print("Warning: Division by zero in aspect ratio calculation. Using fallback.")
            return 4/3
        
        aspect_ratio = np.sqrt(num / den)
        return aspect_ratio
    
    def rectify_image(self, aspect_ratio):
        """
        Rectify the whiteboard image to a rectangular shape.
        
        Args:
            aspect_ratio: The estimated aspect ratio of the whiteboard
        
        Returns:
            Rectified image
        """
        print("Rectifying image...")
        
        # Calculate side lengths of the quadrangle
        W1 = np.linalg.norm(self.corners[1] - self.corners[0])  # top side
        W2 = np.linalg.norm(self.corners[2] - self.corners[3])  # bottom side
        H1 = np.linalg.norm(self.corners[3] - self.corners[0])  # left side
        H2 = np.linalg.norm(self.corners[2] - self.corners[1])  # right side
        
        # Determine maximum dimensions
        W_max = max(W1, W2)
        H_max = max(H1, H2)
        
        # Determine output size based on aspect ratio
        estimated_ratio = W_max / H_max
        if estimated_ratio >= aspect_ratio:
            W_out = int(W_max)
            H_out = int(W_out / aspect_ratio)
        else:
            H_out = int(H_max)
            W_out = int(H_out * aspect_ratio)
        
        # Define source and destination points for perspective transform
        src_points = self.corners.astype(np.float32)
        dst_points = np.array([
            [0, 0],
            [W_out - 1, 0],
            [W_out - 1, H_out - 1],
            [0, H_out - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform
        rectified = cv2.warpPerspective(self.image, M, (W_out, H_out))
        
        return rectified
    
    def enhance_colors(self):
        """
        Enhance colors by making the background uniformly white and increasing
        color saturation of the pen strokes.
        
        Returns:
            Color enhanced image
        """
        print("Enhancing colors...")
        
        image = self.rectified_image.copy()
        height, width = image.shape[:2]
        
        # Convert to float for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Divide the image into cells
        cell_height = self.cell_size
        cell_width = self.cell_size
        rows = int(np.ceil(height / cell_height))
        cols = int(np.ceil(width / cell_width))
        
        # Initialize blank whiteboard image (estimation of C_light)
        blank_whiteboard = np.zeros_like(image_float)
        
        # Process each cell
        for row in range(rows):
            for col in range(cols):
                # Get cell coordinates
                y_start = row * cell_height
                y_end = min((row + 1) * cell_height, height)
                x_start = col * cell_width
                x_end = min((col + 1) * cell_width, width)
                
                # Extract cell
                cell = image_float[y_start:y_end, x_start:x_end]
                
                # Skip empty cells
                if cell.size == 0:
                    continue
                
                # Calculate luminance
                luminance = 0.299 * cell[:,:,0] + 0.587 * cell[:,:,1] + 0.114 * cell[:,:,2]
                
                # Sort pixels by luminance
                flat_luminance = luminance.flatten()
                sorted_indices = np.argsort(flat_luminance)
                
                # Get top 25% brightest pixels
                top_quarter = sorted_indices[int(0.75 * len(sorted_indices)):]
                
                # Calculate average color of brightest pixels
                bright_colors = []
                for idx in top_quarter:
                    y_idx = idx // cell.shape[1]
                    x_idx = idx % cell.shape[1]
                    bright_colors.append(cell[y_idx, x_idx])
                
                if bright_colors:
                    cell_color = np.mean(bright_colors, axis=0)
                else:
                    # Fallback to white if no bright pixels
                    cell_color = np.array([1.0, 1.0, 1.0])
                
                # Set cell color in blank whiteboard
                blank_whiteboard[y_start:y_end, x_start:x_end] = cell_color
        
        # Smooth the blank whiteboard estimation
        blank_whiteboard = cv2.GaussianBlur(blank_whiteboard, (21, 21), 0)
        
        # Normalize each pixel by dividing by the estimated whiteboard color
        normalized = np.zeros_like(image_float)
        for c in range(3):
            channel = image_float[:,:,c]
            bg_channel = blank_whiteboard[:,:,c]
            # Avoid division by zero
            bg_channel = np.maximum(bg_channel, 0.01)
            normalized[:,:,c] = np.minimum(channel / bg_channel, 1.0)
        
        # Apply S-curve to increase contrast and saturation
        p = 0.75  # Controls steepness of S-curve
        enhanced = 0.5 - 0.5 * np.cos(normalized**p * np.pi)
        
        # Convert back to uint8
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def show_results(self):
        """Display the processing results."""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Original Image with Detected Whiteboard")
        plt.imshow(self.display_image)
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title("Rectified Image")
        plt.imshow(self.rectified_image)
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title("Enhanced Image")
        plt.imshow(self.enhanced_image)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_path):
        """Save the processing results."""
        # Convert back to BGR for saving with OpenCV
        result_bgr = cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"Saved result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Whiteboard Digitizer')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--output', '-o', default='whiteboard_output.jpg', help='Output image path')
    parser.add_argument('--manual', '-m', action='store_true', help='Use manual corner selection')
    args = parser.parse_args()
    
    digitizer = WhiteboardDigitizer()
    
    if args.manual:
        # For manual corner selection, we need to display the image
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not read image at {args.input}")
            return
        
        # Resize large images for easier viewing
        height, width = image.shape[:2]
        if max(height, width) > 1000:
            scale = 1000 / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Create a window for corner selection
        cv2.namedWindow('Select Corners')
        corners = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Select Corners', image)
        
        cv2.setMouseCallback('Select Corners', mouse_callback)
        
        print("Click on the four corners of the whiteboard in this order:")
        print("1. Top-left, 2. Top-right, 3. Bottom-right, 4. Bottom-left")
        
        # Display image and wait for corners
        cv2.imshow('Select Corners', image)
        while len(corners) < 4:
            key = cv2.waitKey(100)
            if key == 27:  # ESC key
                print("Corner selection canceled")
                return
        
        cv2.destroyAllWindows()
        
        # Adjust corners for original image size if resized
        if max(height, width) > 1000:
            scale = max(height, width) / 1000
            corners = [[x * scale, y * scale] for x, y in corners]
        
        result = digitizer.process_image(args.input, corners)
    else:
        result = digitizer.process_image(args.input)
    
    digitizer.show_results()
    digitizer.save_results(args.output)


if __name__ == "__main__":
    main()