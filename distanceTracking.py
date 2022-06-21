import sys
import cv2
from cv2 import rectangle
from cv2 import resize
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
#import calibration as calib


# Open both cameras
cap_right = cv2.VideoCapture("Pedestrian_clip_1.mp4")                    
cap_left =  cv2.VideoCapture("Pedestrian_clip_1.mp4")

frame_rate = 120    #Camera frame rate (maximum at 120 fps)

B = 9               #Distance between the cameras [cm]
f = 6               #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]


#Initial values
count = -1
#resolution
W = 1920//3
H = 1080//3

while(True):
    count += 1

    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()
    frame_right = cv2.resize(frame_right,(W,H))
    frame_left = cv2.resize(frame_left,(W,H))
################## CALIBRATION #########################################################

    #frame_right, frame_left = calib.undistorted(frame_right, frame_left)

########################################################################################

    # If cannot catch any frame, break
    if ret_right==False or ret_left==False:                    
        break

    else:
        # APPLYING HSV-FILTER:
        mask_right = hsv.add_HSV_filter(frame_right, 1)
        mask_left = hsv.add_HSV_filter(frame_left, 0)

        # Result-frames after applying HSV-filter mask
        res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
        res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left) 

        # APPLYING SHAPE RECOGNITION:
        circles_right = shape.find_circles(frame_right, mask_right)
        circles_left  = shape.find_circles(frame_left, mask_left)

        # Hough Transforms can be used aswell or some neural network to do object detection


        ################## CALCULATING BALL DEPTH #########################################################

        # If no ball can be caught in one camera show text "TRACKING LOST"
        if np.all(circles_right) == None or np.all(circles_left) == None:
            cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            depth = tri.find_depth(circles_right, circles_left, frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_right, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_left, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print("Depth: ", depth)                                            

        
        #Prikaz frame-a
        cv2.imshow("frame right", frame_right) 
        cv2.imshow("frame left", frame_left)
        #pozicija prozora frame
        cv2.moveWindow('frame right',650,0)
        cv2.moveWindow('frame left',0,0)
        #Prikaz maske
        cv2.imshow("mask right", mask_right) 
        cv2.imshow("mask left", mask_left)
        #pozicija prozora maske
        cv2.moveWindow('mask right',0,400)
        cv2.moveWindow('mask left',650,400)

        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()