# This is the core function for capturing the original faces intended for classification

import cv2
import time
import os

def capture_faces(name, manual=False):
    # The function only requires a name for saving the images
    # and a mode choice
    # Manual Mode - users press c to capture images at their leisure
    # Timer Mode - images will be captured every 3 seconds
    
    epoch = time.time() # Grab time at start of running function
    directory = os.path.join(os.path.abspath("./Dataset/Train"))

    # To ensure images are not overwritten
    # check for the latest image by finding the largest number in the file name
    cap_cnt = -1
    for file in os.listdir(directory):
        if name in file:
            temp = int(file.split(".")[1])
            if temp > cap_cnt:
                cap_cnt = temp
    cap_cnt += 1
    print("previous pictures: ", cap_cnt-1)

    cv2.namedWindow(name)
    
    # Load the haarcascade provided by OpenCV for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    webcam = cv2.VideoCapture(0)

    success = webcam.isOpened()
    if success == False:
        print('Error: Camera could not be opened')
    else:
        print('Success: Grabbed the camera')
        
    while True:
        ret, frame = webcam.read()
        
        frame = cv2.flip(frame,1) # Otherwise video is not mirrored
        
        save_frame = frame.copy()
        
        # Let the user know what mode the camera is in
        mode = "Manual" if manual else "Timer"
        text = f"Capture Mode: {mode}"
        
        cv2.putText(frame, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 3, cv2.LINE_AA)

        # Detect faces in current frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.4, minNeighbors=6)
        for (x,y,w,h) in faces:
            # Loop through all detected faces and grab coordinates of bounding box
            if ((time.time() - epoch) >= 3) and not manual:
                # For timer mode, check if 3 seconds have passed, then capture new image
                cap_name = f"{name}.{cap_cnt}.png"
                p = 0
                # Save only the area of the frame in which a face is found
                cv2.imwrite(os.path.join(directory,cap_name), save_frame[y-p+1:y+h+p, x-p+1:x+w+p])
                print(f"Saving {cap_name}!")
                cap_cnt += 1
                epoch = time.time()
            
            # Draw rectangle around detected faces for user to know when and where a face is detected
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        cv2.imshow(name, frame)
        
        if not ret:
            break
            
        # Monitor keystrokes
        k = cv2.waitKey(1)

        if k & 0xFF == ord('q'):
            # Press q to quit
            print("Quitting...")
            break
        elif (k & 0xFF == ord('c')) and manual:
            # If mode is manual, press c whenever user wants to capture image
            cap_name = f"{name}.{cap_cnt}.png"
            p = 0 # add padding to crop if necessary
            cv2.imwrite(os.path.join(directory,cap_name), save_frame[y-p+1:y+h+p, x-p+1:x+w+p])
            print(f"Saving {cap_name}!")
            cap_cnt += 1

    webcam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
