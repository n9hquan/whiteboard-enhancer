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
        
        # Enhancement parameters
        self.enhancement_params = {
            'denoise_h': tk.IntVar(value=10),
            'clahe_clip': tk.DoubleVar(value=2.0),
            'clahe_grid': tk.IntVar(value=8),
            'adaptive_block': tk.IntVar(value=15),
            'adaptive_c': tk.IntVar(value=9),
            'use_adaptive': tk.BooleanVar(value=True),
            'threshold': tk.IntVar(value=160),
            'blur_size': tk.IntVar(value=3),
            'saturation_boost': tk.DoubleVar(value=1.2)
        }
        
        # Create UI
        self.create_menu()
        self.create_toolbar()
        self.create_main_content()
        self.create_status_bar()
        
        # Settings panel (initially hidden)
        self.settings_panel = None
        
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
        
        # Settings button
        settings_btn = ttk.Button(toolbar_frame, text="Threshold Settings", command=self.toggle_settings_panel)
        settings_btn.pack(side=tk.LEFT, padx=20)
        
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
            # Get current enhancement parameters
            params = {}
            for key, var in self.enhancement_params.items():
                params[key] = var.get()
            
            # Apply enhancement with custom parameters
            self.enhanced_image = enhance_whiteboard(self.warped_image, params)
            self.status_var.set("Image enhanced.")
            
            # Update display
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error enhancing image: {str(e)}")
            self.status_var.set("Error in image enhancement.")
    
    def toggle_settings_panel(self):
        """Toggle the visibility of the settings panel."""
        if self.settings_panel is None or not self.settings_panel.winfo_exists():
            self.create_settings_panel()
        else:
            self.settings_panel.destroy()
            self.settings_panel = None
    
    def create_settings_panel(self):
        """Create a panel with sliders for adjusting enhancement parameters."""
        # Create a new toplevel window
        self.settings_panel = tk.Toplevel(self.root)
        self.settings_panel.title("Fine-tune Enhancement")
        self.settings_panel.geometry("400x500")
        self.settings_panel.transient(self.root)  # Make it float on top of the main window
        
        # Dictionary to store references to all parameter scales
        self.param_scales = {}
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.settings_panel)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas with scrollbar for many controls
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add controls for each parameter
        ttk.Label(scrollable_frame, text="Fine-tune Enhancement", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Thresholding parameters - most important for fine-tuning
        ttk.Label(scrollable_frame, text="Thresholding", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Adaptive threshold checkbox
        adaptive_frame = ttk.Frame(scrollable_frame)
        adaptive_frame.pack(fill="x", pady=5)
        adaptive_check = ttk.Checkbutton(adaptive_frame, text="Use Adaptive Threshold", 
                                       variable=self.enhancement_params['use_adaptive'],
                                       command=self.update_threshold_controls)
        adaptive_check.pack(side="left", padx=5)
        
        # Adaptive threshold parameters
        self.adaptive_frame = ttk.LabelFrame(scrollable_frame, text="Adaptive Threshold Settings")
        self.adaptive_frame.pack(fill="x", pady=5, padx=5)
        self.create_slider(self.adaptive_frame, "Block Size", "adaptive_block", 3, 51, 2)
        self.create_slider(self.adaptive_frame, "C Value", "adaptive_c", -10, 30, 1)
        
        # Global threshold parameter
        self.global_frame = ttk.LabelFrame(scrollable_frame, text="Global Threshold Settings")
        self.global_frame.pack(fill="x", pady=5, padx=5)
        self.create_slider(self.global_frame, "Threshold Value", "threshold", 0, 255, 1)
        
        # Contrast enhancement
        ttk.Separator(scrollable_frame).pack(fill="x", pady=10)
        ttk.Label(scrollable_frame, text="Contrast Enhancement", font=("Arial", 10, "bold")).pack(pady=5)
        self.create_slider(scrollable_frame, "CLAHE Clip Limit", "clahe_clip", 0.5, 5.0, 0.1)
        
        # Final processing
        ttk.Separator(scrollable_frame).pack(fill="x", pady=10)
        ttk.Label(scrollable_frame, text="Final Processing", font=("Arial", 10, "bold")).pack(pady=5)
        self.create_slider(scrollable_frame, "Denoising Strength", "denoise_h", 0, 30, 1)
        self.create_slider(scrollable_frame, "Median Blur Size", "blur_size", 0, 9, 2)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", pady=15)
        
        ttk.Button(button_frame, text="Apply", command=self.enhance_image).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_parameters).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close", command=self.settings_panel.destroy).pack(side="right", padx=5)
        
        # Update threshold controls visibility
        self.update_threshold_controls()
    
    def create_slider(self, parent, label_text, param_key, min_val, max_val, step):
        """Create a labeled slider for a parameter."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        
        ttk.Label(frame, text=label_text).pack(side="top", anchor="w")
        
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill="x")
        
        # For float values, multiply by 10 or 100 to get integer steps for the scale widget
        if isinstance(self.enhancement_params[param_key], tk.DoubleVar):
            # Determine multiplier based on step size
            if step < 0.01:
                multiplier = 1000
            elif step < 0.1:
                multiplier = 100
            else:
                multiplier = 10
                
            scale = ttk.Scale(slider_frame, from_=min_val*multiplier, to=max_val*multiplier, 
                             command=lambda v, pk=param_key, m=multiplier: self.update_param_value(pk, float(v)/m))
            scale.set(self.enhancement_params[param_key].get() * multiplier)
        else:
            scale = ttk.Scale(slider_frame, from_=min_val, to=max_val, 
                             command=lambda v, pk=param_key: self.update_param_value(pk, int(float(v))))
            scale.set(self.enhancement_params[param_key].get())
        
        scale.pack(side="left", fill="x", expand=True)
        
        # Value display
        value_var = tk.StringVar(value=str(self.enhancement_params[param_key].get()))
        value_label = ttk.Label(slider_frame, textvariable=value_var, width=5)
        value_label.pack(side="right", padx=5)
        
        # Store references to update the label when slider moves
        scale.value_var = value_var
        
        # Store reference to the scale in the dictionary
        self.param_scales[param_key] = scale
        
        return scale
    
    def update_param_value(self, param_key, value):
        """Update parameter value and its display."""
        self.enhancement_params[param_key].set(value)
        
        # Store the value in the scale's value_var directly
        # This avoids the need to search through all widgets
        if hasattr(self, 'param_scales') and param_key in self.param_scales:
            scale = self.param_scales[param_key]
            if hasattr(scale, 'value_var'):
                scale.value_var.set(f"{value:.1f}" if isinstance(value, float) else str(value))
    
    def update_threshold_controls(self):
        """Update the visibility of threshold controls based on the adaptive checkbox."""
        if not hasattr(self, 'adaptive_frame') or not hasattr(self, 'global_frame'):
            return
            
        use_adaptive = self.enhancement_params['use_adaptive'].get()
        
        if use_adaptive:
            self.adaptive_frame.pack(fill="x", pady=5, padx=5)
            self.global_frame.pack_forget()
        else:
            self.adaptive_frame.pack_forget()
            self.global_frame.pack(fill="x", pady=5, padx=5)
    
    def reset_parameters(self):
        """Reset all parameters to their default values."""
        default_values = {
            'denoise_h': 10,
            'clahe_clip': 2.0,
            'clahe_grid': 8,
            'adaptive_block': 15,
            'adaptive_c': 9,
            'use_adaptive': True,
            'threshold': 160,
            'blur_size': 3,
            'saturation_boost': 1.2
        }
        
        # Set the values in the variables
        for key, value in default_values.items():
            self.enhancement_params[key].set(value)
        
        # Update all sliders using the param_scales dictionary
        if hasattr(self, 'param_scales'):
            for param_key, scale in self.param_scales.items():
                if param_key in default_values:
                    value = default_values[param_key]
                    
                    # Set the scale value
                    if isinstance(value, float):
                        # For float values, determine the appropriate multiplier
                        if hasattr(scale, 'cget'):
                            # Calculate multiplier based on the scale's range
                            scale_range = float(scale.cget('to')) - float(scale.cget('from_'))
                            if scale_range > 100:
                                multiplier = 100
                            elif scale_range > 10:
                                multiplier = 10
                            else:
                                multiplier = 1
                            scale.set(value * multiplier)
                    else:
                        scale.set(value)
                    
                    # Update the displayed value
                    if hasattr(scale, 'value_var'):
                        scale.value_var.set(f"{value:.1f}" if isinstance(value, float) else str(value))
        
        # Update threshold controls visibility
        self.update_threshold_controls()
    
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
        # Clear the figure completely
        plt.figure(self.fig.number)
        plt.clf()
        
        # Re-create the axes
        self.axes = self.fig.subplots(1, 2)
        self.fig.subplots_adjust(hspace=0.1, wspace=0.1)
        
        # Turn off axes for both subplots
        for ax in self.axes:
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
