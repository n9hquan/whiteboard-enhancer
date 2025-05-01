# Whiteboard Enhancer

A tool to detect, enhance, and export whiteboard photos into clear PDF documents using computer vision.

## Features
- Detect and correct whiteboard perspective
- Enhance contrast and reduce noise
- Convert to high-contrast black & white
- Export as PDF

## Folder Structure

```bash
whiteboard-enhancer/
├── whiteboard/             # Your Python package
│   ├── __init__.py
│   ├── processor.py        # Image processing functions
│   ├── pdf_export.py       # PDF export logic
│   └── utils.py            # Helper functions
├── data/                   # Sample whiteboard images (optional)
├── tests/                  # Unit tests
│   ├── test_processor.py
│   ├── test_pdf_export.py
├── examples/               # Example scripts / notebooks
│   └── demo.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt        # Python dependencies
└── setup.py                # (optional) if you plan to make it pip-installable

## Installation

```bash
pip install -r requirements.txt
