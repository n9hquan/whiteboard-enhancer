import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

from detector import detect_board
from enhancer import enhance_whiteboard
from transformer import four_point_transform
from save_as_pdf import save_as_pdf

class BatchProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whiteboard Batch Processor")
        self.root.geometry("1200x800")
        
        self.images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        self.image_files = self.get_image_files()
        self.current_index = 0
        self.processed_images = []
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Navigation buttons
        ttk.Button(control_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Image info label
        self.info_label = ttk.Label(control_frame, text="")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Save buttons
        ttk.Button(control_frame, text="Save Current as PDF", command=self.save_current_pdf).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Save All as PDF", command=self.save_all_pdf).pack(side=tk.RIGHT, padx=5)
        
        # Create figure for displaying images
        self.figure, self.axs = plt.subplots(1, 4, figsize=(12, 6))
        plt.tight_layout()
        
        # Create canvas for matplotlib figure
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Process all images
        self.process_all_images()
        
        # Display first image
        if self.image_files:
            self.update_display()
    
    def get_image_files(self):
        """Get all image files from the images folder"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return [f for f in os.listdir(self.images_folder) 
                if os.path.isfile(os.path.join(self.images_folder, f)) 
                and os.path.splitext(f)[1].lower() in valid_extensions]
    
    def process_all_images(self):
        """Process all images in the folder"""
        self.processed_images = []
        for img_file in self.image_files:
            try:
                img_path = os.path.join(self.images_folder, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    self.processed_images.append({
                        'filename': img_file,
                        'error': 'Failed to load image'
                    })
                    continue
                
                # Detect whiteboard
                contour = detect_board(image)
                
                if contour is None:
                    self.processed_images.append({
                        'filename': img_file,
                        'original': image,
                        'error': 'Could not detect whiteboard contour'
                    })
                    continue
                
                # Transform and enhance
                warped = four_point_transform(image, contour)
                enhanced = enhance_whiteboard(warped)
                
                # Store results
                self.processed_images.append({
                    'filename': img_file,
                    'original': image,
                    'contour': contour,
                    'warped': warped,
                    'enhanced': enhanced
                })
            except Exception as e:
                self.processed_images.append({
                    'filename': img_file,
                    'error': str(e)
                })
    
    def update_display(self):
        """Update the display with the current image"""
        if not self.image_files:
            messagebox.showinfo("Info", "No images found in the images folder.")
            return
        
        # Clear all subplots
        for ax in self.axs:
            ax.clear()
            ax.axis('off')
        
        current = self.processed_images[self.current_index]
        filename = current['filename']
        
        # Update info label
        self.info_label.config(text=f"Image {self.current_index + 1} of {len(self.image_files)}: {filename}")
        
        if 'error' in current:
            # Display error message
            self.axs[0].text(0.5, 0.5, f"Error: {current['error']}", 
                           ha='center', va='center', fontsize=12, color='red')
            if 'original' in current:
                # Show original if available
                self.axs[0].imshow(cv2.cvtColor(current['original'], cv2.COLOR_BGR2RGB))
                self.axs[0].set_title("Original Image")
        else:
            # Display all processing steps
            self.axs[0].imshow(cv2.cvtColor(current['original'], cv2.COLOR_BGR2RGB))
            self.axs[0].set_title("Original Image")
            
            debug_image = current['original'].copy()
            cv2.drawContours(debug_image, [current['contour']], -1, (0,255,0), 3)
            self.axs[1].imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
            self.axs[1].set_title("Contour Detection")
            
            self.axs[2].imshow(cv2.cvtColor(current['warped'], cv2.COLOR_BGR2RGB))
            self.axs[2].set_title("Perspective Corrected")
            
            self.axs[3].imshow(current['enhanced'], cmap='gray')
            self.axs[3].set_title("Enhanced (Binarized)")
        
        plt.tight_layout()
        self.canvas.draw()
    
    def next_image(self):
        """Go to next image"""
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.update_display()
    
    def prev_image(self):
        """Go to previous image"""
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.update_display()
    
    def save_current_pdf(self):
        """Save current image as PDF"""
        if not self.image_files:
            return
        
        current = self.processed_images[self.current_index]
        if 'error' in current and 'original' not in current:
            messagebox.showerror("Error", f"Cannot save image with error: {current['error']}")
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(self.images_folder), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original
        original_filename = os.path.join(output_dir, f"original_{current['filename']}.pdf")
        save_as_pdf(current['original'], original_filename)
        
        # Save enhanced if available
        if 'enhanced' in current:
            enhanced_filename = os.path.join(output_dir, f"enhanced_{current['filename']}.pdf")
            save_as_pdf(current['enhanced'], enhanced_filename)
            messagebox.showinfo("Saved", f"Saved PDFs to:\n{original_filename}\n{enhanced_filename}")
        else:
            messagebox.showinfo("Saved", f"Saved original PDF to:\n{original_filename}")
    
    def save_all_pdf(self):
        """Save all images as PDFs"""
        if not self.image_files:
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(self.images_folder), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        for img_data in self.processed_images:
            if 'original' in img_data:
                # Save original
                original_filename = os.path.join(output_dir, f"original_{img_data['filename']}.pdf")
                save_as_pdf(img_data['original'], original_filename)
                
                # Save enhanced if available
                if 'enhanced' in img_data:
                    enhanced_filename = os.path.join(output_dir, f"enhanced_{img_data['filename']}.pdf")
                    save_as_pdf(img_data['enhanced'], enhanced_filename)
                
                saved_count += 1
        
        messagebox.showinfo("Saved", f"Saved {saved_count} images as PDFs to:\n{output_dir}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BatchProcessorGUI(root)
    root.mainloop()
