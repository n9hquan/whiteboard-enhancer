import tkinter as tk
from detector import detect_board
from enhancer import enhance_whiteboard
from transformer import four_point_transform
from gui import WhiteboardScannerGUI
from save_as_pdf import save_as_pdf

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Whiteboard Enhancer")
    root.geometry("1000x800")  # Set a reasonable initial size
    
    # Create the application
    app = WhiteboardScannerGUI(root, detect_board, four_point_transform, enhance_whiteboard, save_as_pdf)
    
    # Start the main loop
    root.mainloop()
