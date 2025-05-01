# examples/demo_detect.py

import cv2
import sys
sys.path.append('src')  

from whiteboard.processor import load_image, detect_whiteboard, correct_perspective


def main(image_path):
    # Step 1: Load image
    image = load_image(image_path)

    # Step 2: Detect whiteboard corners
    try:
        corners = detect_whiteboard(image)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Step 3: Draw detected corners on the original image (for visualization)
    for point in corners:
        point = tuple(int(x) for x in point)
        cv2.circle(image, point, 10, (0, 255, 0), -1)

    # Step 4: Correct perspective
    warped = correct_perspective(image, corners)

    # Step 5: Show images
    cv2.imshow("Original with Detected Corners", image)
    cv2.imshow("Warped (Perspective Corrected)", warped)
    print("[INFO] Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/demo_detect.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)
