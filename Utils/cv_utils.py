# This file contains utility functions pertaining to the use of OpenCV
import cv2

def cv_disp_img(img, title="image"):
    # Function for displaying images
    # Expects an image data structure (typically a 2 or 3 rank matrix)
    # and an optional title for the pop-up window
    while True:
        # A loop is needed in order to run properly in JupyterLab
        cv2.imshow(title, img)
        k = cv2.waitKey(1)

        if k & 0xFF == ord('q'):
            # Press q to close the window
            print("Quitting...")
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
