import cv2
import numpy as np
import img2pdf
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

# ----- Image Processing Functions -----

def detect_board(image):
    img_area = image.shape[0] * image.shape[1]

    def whiteboard_detector():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * img_area:
                return approx
        return None

    def smartboard_detector():
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 70])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * img_area:
                return approx
        return None

    contour = whiteboard_detector()
    if contour is not None:
        return contour

    contour = smartboard_detector()
    if contour is not None:
        return contour

    return None

def four_point_transform(image, pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def enhance_whiteboard(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 15, 9)
    enhanced = cv2.medianBlur(enhanced, 3)
    return enhanced

def save_as_pdf(image, filename):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    temp_img = 'temp_save.jpg'
    cv2.imwrite(temp_img, image)

    with open(filename, "wb") as f:
        f.write(img2pdf.convert(temp_img))

    os.remove(temp_img)

# ----- GUI Functions -----

class WhiteboardScannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whiteboard Scanner to PDF")
        self.image = None
        self.contour = None
        self.warped = None
        self.enhanced = None

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
        self.contour = detect_board(self.image)

        if self.contour is None:
            messagebox.showerror("Error", "Could not detect whiteboard contour.")
            return

        self.warped = four_point_transform(self.image, self.contour)
        self.enhanced = enhance_whiteboard(self.warped)
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
            save_as_pdf(self.warped, file_path)
            messagebox.showinfo("Saved", f"Color PDF saved at:\n{file_path}")

    def save_enhanced_pdf(self):
        if self.enhanced is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            save_as_pdf(self.enhanced, file_path)
            messagebox.showinfo("Saved", f"Enhanced PDF saved at:\n{file_path}")

# ----- Main -----
if __name__ == "__main__":
    root = tk.Tk()
    app = WhiteboardScannerGUI(root)
    root.mainloop()
