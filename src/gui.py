import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageTk

class WhiteboardScannerGUI:
    def __init__(self, root, detect_board, four_point_transform, enhance_whiteboard, save_as_pdf):
        self.root = root
        self.root.title("Whiteboard Scanner to PDF")
        self.image = None
        self.contour = None
        self.warped = None
        self.enhanced = None
        self.manual_corners = []
        self.manual_mode = False
        self.corner_index = 0
        self.confidence_threshold = 0.6  # Threshold for automatic vs manual detection

        self.detect_board = detect_board
        self.four_point_transform = four_point_transform
        self.enhance_whiteboard = enhance_whiteboard
        self.save_as_pdf = save_as_pdf

        # Create control frame
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        
        tk.Button(control_frame, text="Browse Image", command=self.browse_image, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Save Color PDF", command=self.save_color_pdf, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Save Enhanced PDF", command=self.save_enhanced_pdf, width=20).pack(side=tk.LEFT, padx=5)
        
        # Manual corner selection frame
        self.manual_frame = tk.Frame(root)
        self.manual_frame.pack(pady=5)
        
        self.manual_label = tk.Label(self.manual_frame, text="Manual corner selection mode: Inactive", fg="gray")
        self.manual_label.pack(side=tk.LEFT, padx=5)
        
        self.manual_button = tk.Button(self.manual_frame, text="Enable Manual Selection", command=self.toggle_manual_mode, width=20)
        self.manual_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(self.manual_frame, text="Reset Corners", command=self.reset_corners, width=15, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=5)

        # Create matplotlib figure
        self.figure, self.axs = plt.subplots(2, 2, figsize=(10,8))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()
        
        # Connect the click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Reset manual corners and mode
        self.reset_corners()
        self.manual_mode = False
        self.manual_button.config(text="Enable Manual Selection")
        self.manual_label.config(text="Manual corner selection mode: Inactive", fg="gray")
        
        try:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
                
            # Try automatic detection with confidence score
            self.contour, confidence = self.detect_board(self.image)
            
            # Update status with confidence information
            confidence_percent = int(confidence * 100)
            self.status_label.config(text=f"Detection confidence: {confidence_percent}%")
            
            if confidence < self.confidence_threshold:
                # Low confidence, suggest manual selection
                response = messagebox.askquestion("Low Confidence Detection", 
                                               f"Automatic detection confidence is low ({confidence_percent}%). \n"
                                               f"Would you like to manually select the corners?")
                if response == 'yes':
                    self.toggle_manual_mode()
                    return
            
            # Process the image with detected contour
            self.process_with_contour()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def process_with_contour(self):
        """Process the image with the current contour (auto or manual)"""
        try:
            if self.contour is None or len(self.contour) != 4:
                messagebox.showerror("Error", "Invalid contour. Please try again or use manual selection.")
                return
                
            self.warped = self.four_point_transform(self.image, self.contour)
            self.enhanced = self.enhance_whiteboard(self.warped)
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")

    def update_display(self):
        if self.image is None:
            return
            
        # Clear all subplots
        for ax in self.axs.flat:
            ax.clear()
            ax.axis('off')
        
        # Original image
        self.axs[0,0].set_title("Original Image")
        self.axs[0,0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        # Contour detection
        debug_image = self.image.copy()
        
        if self.manual_mode:
            # Define corner labels and colors
            corner_labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
            corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]  # Blue, Green, Red, Magenta
            
            # If we have corners selected, draw them
            if len(self.manual_corners) > 0:
                # Draw the manual corners that have been selected so far
                for i, corner in enumerate(self.manual_corners):
                    # Convert BGR color to RGB for OpenCV
                    color = corner_colors[i]
                    # Draw a filled circle at the corner
                    cv2.circle(debug_image, tuple(corner), 10, color, -1)
                    # Add the corner number and label
                    cv2.putText(debug_image, corner_labels[i], (corner[0]+10, corner[1]+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw lines between corners
                if len(self.manual_corners) > 1:
                    for i in range(len(self.manual_corners)):
                        pt1 = tuple(self.manual_corners[i])
                        pt2 = tuple(self.manual_corners[(i+1) % len(self.manual_corners)])
                        # Use a gradient color for the lines to show direction
                        color1 = corner_colors[i]
                        color2 = corner_colors[(i+1) % len(self.manual_corners)]
                        # Average the colors for the line
                        line_color = ((color1[0] + color2[0])//2, 
                                     (color1[1] + color2[1])//2, 
                                     (color1[2] + color2[2])//2)
                        cv2.line(debug_image, pt1, pt2, line_color, 2)
            
            # If we don't have all corners yet, show the expected positions
            if len(self.manual_corners) < 4:
                # Add semi-transparent overlay to show where corners should be
                overlay = debug_image.copy()
                h, w = debug_image.shape[:2]
                # Draw expected corner positions with transparency
                expected_corners = [(int(w*0.1), int(h*0.1)),  # Top-Left
                                  (int(w*0.9), int(h*0.1)),  # Top-Right
                                  (int(w*0.9), int(h*0.9)),  # Bottom-Right
                                  (int(w*0.1), int(h*0.9))]  # Bottom-Left
                
                # Only show corners that haven't been selected yet
                for i in range(len(self.manual_corners), 4):
                    cv2.circle(overlay, expected_corners[i], 15, corner_colors[i], 2)
                    cv2.putText(overlay, f"Expected {i+1}", 
                              (expected_corners[i][0]+15, expected_corners[i][1]+15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, corner_colors[i], 1)
                
                # Apply the overlay with transparency
                cv2.addWeighted(overlay, 0.3, debug_image, 0.7, 0, debug_image)
                
            title = f"Manual Selection ({len(self.manual_corners)}/4) - Select in Clockwise Order"
        else:
            # Draw the detected contour
            if self.contour is not None:
                cv2.drawContours(debug_image, [self.contour], -1, (0, 255, 0), 3)
            title = "Contour Detection"
        
        self.axs[0,1].set_title(title)
        self.axs[0,1].imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        
        # Only show warped and enhanced if they exist
        if self.warped is not None:
            self.axs[1,0].set_title("Perspective Corrected")
            self.axs[1,0].imshow(cv2.cvtColor(self.warped, cv2.COLOR_BGR2RGB))
        
        if self.enhanced is not None:
            self.axs[1,1].set_title("Enhanced (Binarized)")
            self.axs[1,1].imshow(self.enhanced, cmap='gray')
        
        self.canvas.draw()
        
    def toggle_manual_mode(self):
        """Toggle between automatic and manual corner selection"""
        self.manual_mode = not self.manual_mode
        
        if self.manual_mode:
            self.manual_button.config(text="Apply Manual Selection")
            self.manual_label.config(text="Manual corner selection mode: ACTIVE - Select corners in clockwise order", fg="red")
            self.reset_button.config(state=tk.NORMAL)
            self.reset_corners()
            
            # Show corner selection guide
            self.show_corner_selection_guide()
            
            # Open the separate window for corner selection
            self.open_corner_selection_window()
        else:
            # If we have 4 corners, apply the transformation
            if len(self.manual_corners) == 4:
                # Convert to numpy array for the contour
                self.contour = np.array(self.manual_corners, dtype=np.int32)
                self.process_with_contour()
                
            self.manual_button.config(text="Enable Manual Selection")
            self.manual_label.config(text="Manual corner selection mode: Inactive", fg="gray")
            self.reset_button.config(state=tk.DISABLED)
            
            # Close the corner selection window if it exists
            if hasattr(self, 'corner_window') and self.corner_window is not None:
                self.corner_window.destroy()
                self.corner_window = None
    
    def open_corner_selection_window(self):
        """Open a separate window for corner selection"""
        if self.image is None:
            return
            
        # Create a new top-level window
        self.corner_window = tk.Toplevel(self.root)
        self.corner_window.title("Select Corners - Clockwise from Top-Left")
        self.corner_window.protocol("WM_DELETE_WINDOW", self.on_corner_window_close)
        
        # Create frame for instructions
        instruction_frame = tk.Frame(self.corner_window)
        instruction_frame.pack(pady=10)
        
        # Add instructions
        instructions = (
            "Select the 4 corners of the whiteboard in CLOCKWISE order:\n"
            "1. Top-Left (Blue)\n"
            "2. Top-Right (Green)\n"
            "3. Bottom-Right (Red)\n"
            "4. Bottom-Left (Magenta)"
        )
        tk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack()
        
        # Create frame for the image
        self.corner_canvas_frame = tk.Frame(self.corner_window)
        self.corner_canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Resize image to fit the window while maintaining aspect ratio
        max_width, max_height = 800, 600
        h, w = self.image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_width, new_height = int(w * scale), int(h * scale)
        
        # Resize the image for display
        self.selection_image = cv2.resize(self.image.copy(), (new_width, new_height))
        self.display_image = self.selection_image.copy()
        
        # Convert to PIL format for Tkinter
        self.photo = self.convert_cv_to_pil(self.display_image)
        
        # Create canvas for the image
        self.corner_canvas = tk.Canvas(self.corner_canvas_frame, width=new_width, height=new_height)
        self.corner_canvas.pack()
        
        # Display the image on the canvas
        self.corner_image_on_canvas = self.corner_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Bind mouse events
        self.corner_canvas.bind("<Button-1>", self.on_corner_canvas_click)
        
        # Create button frame
        button_frame = tk.Frame(self.corner_window)
        button_frame.pack(pady=10)
        
        # Add buttons
        tk.Button(button_frame, text="Reset Corners", command=self.reset_corners).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Apply Selection", command=self.apply_corner_selection).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.corner_status = tk.Label(self.corner_window, text="Select corner 1 (Top-Left)", fg="blue")
        self.corner_status.pack(pady=5)
        
        # Center the window
        self.corner_window.update_idletasks()
        width = self.corner_window.winfo_width()
        height = self.corner_window.winfo_height()
        x = (self.corner_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.corner_window.winfo_screenheight() // 2) - (height // 2)
        self.corner_window.geometry(f"{width}x{height}+{x}+{y}")
    
    def convert_cv_to_pil(self, cv_image):
        """Convert OpenCV image to PIL format for Tkinter"""
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        # Convert to PhotoImage
        return ImageTk.PhotoImage(pil_image)
    
    def on_corner_canvas_click(self, event):
        """Handle clicks on the corner selection canvas"""
        if len(self.manual_corners) >= 4:
            self.corner_status.config(text="All corners selected. Click 'Apply Selection' to continue.")
            return
            
        # Get the click coordinates
        x, y = event.x, event.y
        
        # Scale back to original image coordinates
        orig_h, orig_w = self.image.shape[:2]
        disp_h, disp_w = self.selection_image.shape[:2]
        orig_x = int(x * (orig_w / disp_w))
        orig_y = int(y * (orig_h / disp_h))
        
        # Store the corner in original image coordinates
        self.manual_corners.append([orig_x, orig_y])
        
        # Update the display
        self.update_corner_selection_display()
        
        # Update status
        corner_num = len(self.manual_corners)
        if corner_num < 4:
            corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            corner_colors = ["blue", "green", "red", "magenta"]
            self.corner_status.config(
                text=f"Select corner {corner_num+1} ({corner_names[corner_num]})", 
                fg=corner_colors[corner_num]
            )
        else:
            self.corner_status.config(text="All corners selected. Click 'Apply Selection' to continue.", fg="black")
    
    def update_corner_selection_display(self):
        """Update the corner selection display"""
        if not hasattr(self, 'selection_image') or self.selection_image is None:
            return
            
        # Make a copy of the original image
        display_img = self.selection_image.copy()
        
        # Define corner colors (BGR format for OpenCV)
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]  # Blue, Green, Red, Magenta
        corner_names = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
        
        # Scale coordinates to display image size
        scaled_corners = []
        if self.manual_corners:
            orig_h, orig_w = self.image.shape[:2]
            disp_h, disp_w = display_img.shape[:2]
            
            for corner in self.manual_corners:
                x = int(corner[0] * (disp_w / orig_w))
                y = int(corner[1] * (disp_h / orig_h))
                scaled_corners.append((x, y))
        
        # Draw the corners and lines
        for i, corner in enumerate(scaled_corners):
            # Draw circle at corner
            cv2.circle(display_img, corner, 10, corner_colors[i], -1)
            # Add label
            cv2.putText(display_img, corner_names[i], 
                      (corner[0] + 10, corner[1] + 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, corner_colors[i], 2)
        
        # Draw lines between corners
        if len(scaled_corners) > 1:
            for i in range(len(scaled_corners)):
                pt1 = scaled_corners[i]
                pt2 = scaled_corners[(i+1) % len(scaled_corners)]
                # Draw arrow to show direction
                cv2.arrowedLine(display_img, pt1, pt2, corner_colors[i], 2, tipLength=0.03)
        
        # Update the image on the canvas
        self.photo = self.convert_cv_to_pil(display_img)
        self.corner_canvas.itemconfig(self.corner_image_on_canvas, image=self.photo)
    
    def apply_corner_selection(self):
        """Apply the manual corner selection"""
        if len(self.manual_corners) != 4:
            messagebox.showwarning("Incomplete Selection", 
                                "Please select all 4 corners before applying.")
            return
            
        # Convert to numpy array for the contour
        self.contour = np.array(self.manual_corners, dtype=np.int32)
        
        # Close the corner selection window
        if self.corner_window is not None:
            self.corner_window.destroy()
            self.corner_window = None
        
        # Process the image with the selected contour
        self.process_with_contour()
        
        # Update the main GUI
        self.manual_mode = False
        self.manual_button.config(text="Enable Manual Selection")
        self.manual_label.config(text="Manual corner selection mode: Inactive", fg="gray")
        self.reset_button.config(state=tk.DISABLED)
    
    def on_corner_window_close(self):
        """Handle corner selection window close event"""
        # Reset manual mode
        self.manual_mode = False
        self.manual_button.config(text="Enable Manual Selection")
        self.manual_label.config(text="Manual corner selection mode: Inactive", fg="gray")
        self.reset_button.config(state=tk.DISABLED)
        
        # Close the window
        self.corner_window.destroy()
        self.corner_window = None
            
    def show_corner_selection_guide(self):
        """Show a message box explaining the corner selection order"""
        messagebox.showinfo("Corner Selection Guide", 
                          "Please select the 4 corners of the whiteboard in CLOCKWISE order:\n\n"
                          "1. Top-Left\n"
                          "2. Top-Right\n"
                          "3. Bottom-Right\n"
                          "4. Bottom-Left\n\n"
                          "This order is important for correct perspective transformation.")

    
    def reset_corners(self):
        """Reset the manually selected corners"""
        self.manual_corners = []
        self.corner_index = 0
        if self.image is not None:
            self.update_display()
    
    def on_click(self, event):
        """Handle mouse clicks for manual corner selection"""
        if not self.manual_mode or self.image is None:
            return
            
        # Check if click is in the original image subplot
        if event.inaxes == self.axs[0,0] or event.inaxes == self.axs[0,1]:
            # Convert from display coordinates to image coordinates
            height, width = self.image.shape[:2]
            x = int(event.xdata * width / self.axs[0,0].get_xlim()[1])
            y = int(event.ydata * height / self.axs[0,0].get_ylim()[1])
            
            # Add the corner
            if len(self.manual_corners) < 4:
                self.manual_corners.append([x, y])
                self.corner_index += 1
                self.update_display()
                
                # If we have all 4 corners, ask if user wants to apply
                if len(self.manual_corners) == 4:
                    self.manual_button.config(text="Apply Manual Selection")
            else:
                messagebox.showinfo("Info", "You've already selected 4 corners. Click 'Apply Manual Selection' to proceed or 'Reset Corners' to start over.")


    def save_color_pdf(self):
        if self.warped is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.save_as_pdf(self.warped, file_path)
            messagebox.showinfo("Saved", f"Color PDF saved at:\n{file_path}")

    def save_enhanced_pdf(self):
        if self.enhanced is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.save_as_pdf(self.enhanced, file_path)
            messagebox.showinfo("Saved", f"Enhanced PDF saved at:\n{file_path}")
