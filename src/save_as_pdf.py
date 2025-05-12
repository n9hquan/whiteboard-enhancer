# save_as_pdf.py
import cv2
import img2pdf
import os

def save_as_pdf(image, filename):
    """
    Save an image as a PDF file.
    
    Parameters:
    image: Input image (either grayscale or color).
    filename: The name of the output PDF file.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    temp_img = 'temp_save.jpg'
    cv2.imwrite(temp_img, image)

    with open(filename, "wb") as f:
        f.write(img2pdf.convert(temp_img))

    os.remove(temp_img)
