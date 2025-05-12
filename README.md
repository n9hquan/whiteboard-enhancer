# Whiteboard Enhancer

A Python application for scanning, enhancing, and converting whiteboard images to PDF.

## Features

- Automatic whiteboard/smartboard detection
- Perspective correction for top-down view
- Image enhancement for better readability
- PDF conversion
- Multiple detection modes: auto, whiteboard, and smartboard

## Installation

Create and activate a virtual environment, then install the requirements:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Make sure to activate the virtual environment before running:

```bash
# Activate the virtual environment (if not already activated)
source venv/bin/activate

# Run the command-line application with an image file
python main.py path/to/your/image.jpg

# Optional arguments
python main.py path/to/your/image.jpg --mode auto|whiteboard|smartboard --no-display
```

### Graphical User Interface

For a more user-friendly experience, use the GUI application:

```bash
# Activate the virtual environment (if not already activated)
source venv/bin/activate

# Run the GUI application
python gui_app.py
```

The GUI application provides:
- File browsing to select images from your local machine
- Options to switch between color and binary display modes
- Interactive processing with visual feedback
- Easy PDF saving options

### Using as a Library

You can also use individual components:

```python
from whiteboard_enhancer.detector import detect_board
from whiteboard_enhancer.transformer import four_point_transform
from whiteboard_enhancer.enhancer import enhance_whiteboard
from whiteboard_enhancer.utils import save_as_pdf

# Your code here
```

## Project Structure

- `main.py`: Entry point for the application
- `whiteboard_enhancer/`: Main package
  - `detector.py`: Board detection algorithms
  - `transformer.py`: Perspective transformation
  - `enhancer.py`: Image enhancement
  - `utils.py`: Utility functions
  - `ui.py`: User interface components
