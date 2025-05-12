import tkinter as tk
from detector import detect_board
from enhancer import enhance_whiteboard
from transformer import four_point_transform
from gui import WhiteboardScannerGUI
from save_as_pdf import save_as_pdf

if __name__ == "__main__":
    root = tk.Tk()
    app = WhiteboardScannerGUI(root, detect_board, four_point_transform, enhance_whiteboard, save_as_pdf)
    root.mainloop()
