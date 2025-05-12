"""
GUI application for the whiteboard enhancer.
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
import threading

from whiteboard_enhancer.detector import detect_board
from whiteboard_enhancer.transformer import four_point_transform
from whiteboard_enhancer.enhancer import enhance_whiteboard
from whiteboard_enhancer.utils import save_as_pdf


class WhiteboardEnhancerApp:
    """Main GUI application for the whiteboard enhancer."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("Whiteboard Enhancer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.contour = None
        self.warped_image = None
        self.enhanced_image = None
        self.mode_var = tk.StringVar(value="auto")
        self.display_mode_var = tk.StringVar(value="color")
        self.result_type_var = tk.StringVar(value="enhanced")
        
        # Create UI
        self.create_menu()
        self.create_toolbar()
        self.create_main_content()
        self.create_status_bar()
        
        # Initial state
        self.update_ui_state(False)
    
    def create_menu(self):
        """Create the menu bar."""
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.browse_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save Color PDF", command=lambda: self.save_pdf(False))
        file_menu.add_command(label="Save Enhanced PDF", command=lambda: self.save_pdf(True))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Process menu
        process_menu = tk.Menu(menu_bar, tearoff=0)
        process_menu.add_command(label="Detect Board", command=self.detect_board)
        process_menu.add_command(label="Enhance", command=self.enhance_image)
        process_menu.add_command(label="Process All", command=self.process_all)
        menu_bar.add_cascade(label="Process", menu=process_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def create_toolbar(self):
        """Create the toolbar with processing options."""
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Browse button
        browse_btn = ttk.Button(toolbar_frame, text="Browse Image", command=self.browse_image)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Display mode
        ttk.Label(toolbar_frame, text="Display Mode:").pack(side=tk.LEFT, padx=(20, 5))
        display_combo = ttk.Combobox(toolbar_frame, textvariable=self.display_mode_var, 
                                    values=["color", "binary"],
                                    width=10, state="readonly")
        display_combo.pack(side=tk.LEFT, padx=5)
        display_combo.bind("<<ComboboxSelected>>", lambda e: self.update_display())
        
        # Show all steps button
        show_steps_btn = ttk.Button(toolbar_frame, text="Show All Steps", command=self.show_all_steps)
        show_steps_btn.pack(side=tk.RIGHT, padx=5)
        
        # Process button
        process_btn = ttk.Button(toolbar_frame, text="Process Image", command=self.process_all)
        process_btn.pack(side=tk.RIGHT, padx=5)
    
    def create_main_content(self):
        """Create the main content area with image display."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Image display area (1x2 grid - original and result)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.subplots_adjust(hspace=0.1, wspace=0.1)
        
        # Set up the initial empty plots with titles
        self.axes[0].set_title("Original Image")
        self.axes[1].set_title("Result")
        
        for ax in self.axes:
            ax.axis('off')
            # Set a gray background for empty plots
            ax.imshow(np.ones((10, 10, 3)) * 0.8, cmap='gray')
            ax.text(0.5, 0.5, "No Image", ha='center', va='center', fontsize=12)
        
        # Embed the matplotlib figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_image(self):
        """Open a file dialog to browse for an image."""
        file_path = filedialog.askopenfilename(
            title="Select Whiteboard Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                messagebox.showerror("Error", "Could not read the image file.")
                return
            
            # Reset processing state
            self.contour = None
            self.warped_image = None
            self.enhanced_image = None
            
            # Update UI
            self.update_ui_state(True)
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
            
            # Display original image
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def detect_board(self):
        """Detect the whiteboard in the image."""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        
        self.status_var.set("Detecting board...")
        self.root.update()
        
        try:
            mode = self.mode_var.get()
            self.contour = detect_board(self.original_image, mode)
            
            if self.contour is None:
                messagebox.showwarning("Warning", "Could not detect whiteboard edges.")
                self.status_var.set("Board detection failed.")
                return
            
            # Transform the image
            self.warped_image = four_point_transform(self.original_image, self.contour)
            self.status_var.set("Board detected and transformed.")
            
            # Update display
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting board: {str(e)}")
            self.status_var.set("Error in board detection.")
    
    def enhance_image(self):
        """Enhance the warped image."""
        if self.warped_image is None:
            messagebox.showinfo("Info", "Please detect the board first.")
            return
        
        self.status_var.set("Enhancing image...")
        self.root.update()
        
        try:
            self.enhanced_image = enhance_whiteboard(self.warped_image)
            self.status_var.set("Image enhanced.")
            
            # Update display
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error enhancing image: {str(e)}")
            self.status_var.set("Error in image enhancement.")
    
    def process_all(self):
        """Process the image through all steps."""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        
        # Run in a separate thread to keep UI responsive
        threading.Thread(target=self._process_all_thread, daemon=True).start()
    
    def _process_all_thread(self):
        """Background thread for processing."""
        try:
            self.status_var.set("Detecting board...")
            self.root.update()
            
            mode = self.mode_var.get()
            self.contour = detect_board(self.original_image, mode)
            
            if self.contour is None:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "Could not detect whiteboard edges."))
                self.status_var.set("Board detection failed.")
                return
            
            self.status_var.set("Transforming image...")
            self.root.update()
            self.warped_image = four_point_transform(self.original_image, self.contour)
            
            self.status_var.set("Enhancing image...")
            self.root.update()
            self.enhanced_image = enhance_whiteboard(self.warped_image)
            
            self.status_var.set("Processing complete.")
            
            # Update display in the main thread
            self.root.after(0, self.update_display)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image: {str(e)}"))
            self.status_var.set("Error in processing.")
    
    def update_display(self):
        """Update the image display based on current state."""
        # Clear all axes
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
        
        # Set titles
        self.axes[0].set_title("Original Image")
        
        # Get display settings
        display_mode = self.display_mode_var.get()
        
        # Original image
        if self.original_image is not None:
            self.axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        else:
            self.axes[0].imshow(np.ones((10, 10, 3)) * 0.8)
            self.axes[0].text(0.5, 0.5, "No Image", ha='center', va='center', fontsize=12)
        
        # Result image
        if self.warped_image is not None:
            if display_mode == "color":
                # Show raw color of the cropped whiteboard
                self.axes[1].set_title("Cropped Whiteboard (Raw Color)")
                self.axes[1].imshow(cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2RGB))
            else:  # binary
                # Show enhanced binary version
                if self.enhanced_image is not None:
                    self.axes[1].set_title("Enhanced Binary Result")
                    self.axes[1].imshow(self.enhanced_image, cmap='gray')
                else:
                    # If enhanced image is not available, show warped anyway
                    self.axes[1].set_title("Cropped Whiteboard")
                    self.axes[1].imshow(cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2RGB))
        else:
            self.axes[1].set_title("Result")
            self.axes[1].imshow(np.ones((10, 10, 3)) * 0.8)
            self.axes[1].text(0.5, 0.5, "Not processed", ha='center', va='center', fontsize=12)
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def show_all_steps(self):
        """Show all processing steps in a separate window."""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        
        # Create a new figure with all processing steps
        plt.figure(figsize=(12, 10))
        plt.suptitle("All Processing Steps", fontsize=16)
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Contour detection
        plt.subplot(2, 2, 2)
        plt.title("Contour Detection")
        if self.contour is not None:
            debug_image = self.original_image.copy()
            cv2.drawContours(debug_image, [self.contour], -1, (0, 255, 0), 3)
            plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            plt.text(0.5, 0.5, "No contour detected", 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=12, color='red')
        plt.axis('off')
        
        # Warped image
        plt.subplot(2, 2, 3)
        plt.title("Perspective Correction")
        if self.warped_image is not None:
            plt.imshow(cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(np.ones((10, 10, 3)) * 0.8)
            plt.text(0.5, 0.5, "Not processed", 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=12)
        plt.axis('off')
        
        # Enhanced image
        plt.subplot(2, 2, 4)
        plt.title("Enhanced Image")
        if self.enhanced_image is not None:
            display_mode = self.display_mode_var.get()
            if display_mode == "binary":
                plt.imshow(self.enhanced_image, cmap='gray')
            else:  # color
                color_enhanced = cv2.cvtColor(self.enhanced_image, cv2.COLOR_GRAY2BGR)
                plt.imshow(cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(np.ones((10, 10, 3)) * 0.8)
            plt.text(0.5, 0.5, "Not processed", 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        plt.show()
    
    def save_pdf(self, enhanced=False):
        """Save the processed image as PDF."""
        if enhanced and self.enhanced_image is None:
            messagebox.showinfo("Info", "Please enhance the image first.")
            return
        elif not enhanced and self.warped_image is None:
            messagebox.showinfo("Info", "Please process the image first.")
            return
        
        # Get save path
        default_name = os.path.splitext(os.path.basename(self.image_path))[0]
        if enhanced:
            default_name = f"enhanced_{default_name}"
        
        save_path = filedialog.asksaveasfilename(
            title="Save PDF",
            defaultextension=".pdf",
            initialfile=f"{default_name}.pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            self.status_var.set("Saving PDF...")
            self.root.update()
            
            image_to_save = self.enhanced_image if enhanced else self.warped_image
            
            # If enhanced image is grayscale, convert to BGR for saving
            if enhanced and len(image_to_save.shape) == 2:
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_GRAY2BGR)
            
            # Save as PDF
            save_as_pdf(image_to_save, save_path)
            
            self.status_var.set(f"PDF saved to: {save_path}")
            messagebox.showinfo("Success", f"PDF saved successfully to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving PDF: {str(e)}")
            self.status_var.set("Error saving PDF.")
    
    def update_ui_state(self, has_image):
        """Update the UI state based on whether an image is loaded."""
        # In some Tkinter versions, the 'state' option may not be supported for menu items
        # So we'll use a try-except block to handle this gracefully
        try:
            state = "normal" if has_image else "disabled"
            
            # Update menu items
            menu = self.root.nametowidget(self.root["menu"])
            
            # Process menu
            process_menu = menu.nametowidget(menu.entrycget(1, "menu"))
            for i in range(3):  # Process menu items
                try:
                    process_menu.entryconfig(i, state=state)
                except tk.TclError:
                    # If state option is not supported, we'll just skip it
                    pass
            
            # File menu
            file_menu = menu.nametowidget(menu.entrycget(0, "menu"))
            for i in range(2, 4):  # Save PDF menu items
                try:
                    file_menu.entryconfig(i, state=state)
                except tk.TclError:
                    # If state option is not supported, we'll just skip it
                    pass
        except Exception:
            # If there's any error with menu configuration, just ignore it
            # This ensures the application can still run even if menu state can't be updated
            pass
    
    def show_about(self):
        """Show the about dialog."""
        messagebox.showinfo(
            "About Whiteboard Enhancer",
            "Whiteboard Enhancer v1.0\n\n"
            "A tool for scanning, enhancing, and converting whiteboard images to PDF.\n\n"
            "Features:\n"
            "- Automatic whiteboard/smartboard detection\n"
            "- Perspective correction\n"
            "- Image enhancement\n"
            "- PDF conversion\n\n"
            "© 2025 Whiteboard Enhancer Team"
        )


def run_gui():
    """Run the GUI application."""
    root = tk.Tk()
    app = WhiteboardEnhancerApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
