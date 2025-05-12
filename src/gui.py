import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2

class WhiteboardScannerGUI:
    def __init__(self, root, detect_board, four_point_transform, enhance_whiteboard, save_as_pdf):
        self.root = root
        self.root.title("Whiteboard Scanner to PDF")
        self.image = None
        self.contour = None
        self.warped = None
        self.enhanced = None

        self.detect_board = detect_board
        self.four_point_transform = four_point_transform
        self.enhance_whiteboard = enhance_whiteboard
        self.save_as_pdf = save_as_pdf

        tk.Button(root, text="Browse Image", command=self.browse_image, width=20).pack(pady=5)
        tk.Button(root, text="Save Color PDF", command=self.save_color_pdf, width=20).pack(pady=5)
        tk.Button(root, text="Save Enhanced PDF", command=self.save_enhanced_pdf, width=20).pack(pady=5)

        self.figure, self.axs = plt.subplots(2, 2, figsize=(10,8))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.image = cv2.imread(file_path)
        self.contour = self.detect_board(self.image)

        if self.contour is None:
            messagebox.showerror("Error", "Could not detect whiteboard contour.")
            return

        self.warped = self.four_point_transform(self.image, self.contour)
        self.enhanced = self.enhance_whiteboard(self.warped)
        self.update_display()

    def update_display(self):
        self.axs[0,0].clear()
        self.axs[0,0].set_title("Original Image")
        self.axs[0,0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.axs[0,0].axis('off')

        debug_image = self.image.copy()
        cv2.drawContours(debug_image, [self.contour], -1, (0,255,0), 3)
        self.axs[0,1].clear()
        self.axs[0,1].set_title("Contour Detection")
        self.axs[0,1].imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        self.axs[0,1].axis('off')

        self.axs[1,0].clear()
        self.axs[1,0].set_title("Perspective Corrected")
        self.axs[1,0].imshow(cv2.cvtColor(self.warped, cv2.COLOR_BGR2RGB))
        self.axs[1,0].axis('off')

        self.axs[1,1].clear()
        self.axs[1,1].set_title("Enhanced (Binarized)")
        self.axs[1,1].imshow(self.enhanced, cmap='gray')
        self.axs[1,1].axis('off')

        self.canvas.draw()

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
