# This class is used for cropping images based off of https://www.life2coding.com/crop-image-using-mouse-click-movement-python/
# A class structure is necessary since several internal methods need to share multiple variables internally

import cv2
import os

class image_cropper():
    # The initial object only requires the path of an image to be croppped
    
    def __init__(self, imagepath):
        self.cropping = False
        self.path = imagepath
        self.roi = []

        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0

        self.image = cv2.imread(imagepath)
        self.oriImage = self.image.copy()

    def mouse_crop(self, event, x, y, flags, param):
        # Use of OpenCV mouse click events is utilized here

        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
            self.cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping == True:
                self.x_end, self.y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            self.x_end, self.y_end = x, y
            self.cropping = False # cropping is finished

            refPoint = [(self.x_start, self.y_start), (self.x_end, self.y_end)]

            if len(refPoint) == 2: #when two points were found
                self.roi = self.oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                try:
                    cv2.imshow("Cropped", self.roi)
                except:
                    pass

    def capture(self):
        # Similar to the displaying of any image using OpenCV in JupyterLab
        # They need to be held in a continuous loop
        
        cv2.namedWindow(self.path)
        cv2.setMouseCallback(self.path, self.mouse_crop)

        while True:

            i = self.image.copy()

            if not self.cropping:
                cv2.imshow(self.path, self.image)

            elif self.cropping:
                cv2.rectangle(i, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                cv2.imshow(self.path, i)

            k = cv2.waitKey(1) & 0xFF

            if k == ord('s'):
                # Press s to save an image to update any cropping performed
                try:
                    cv2.imwrite(self.path, self.roi)
                except:
                    pass
            if k == ord('d'):
                # Press d to delete the image
                try:
                    os.remove(self.path)
                except:
                    pass
            if k == ord('n'):
                # Press n to ignore image or move onto next image after saving a crop
                break
            if k == ord('q'):
                # Press q to quit early if cropping is being done by looping over a set of images
                return 1

        # close all open windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)
