from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QAbstractButton
import numpy as np
import cv2 
import imutils
import time



class Widget(QWidget):
    def __init__(self):
        super().__init__()

        reset_button = QPushButton("Show webcam")
        reset_button.clicked.connect(self.show_webcam)

        contour_button = QPushButton("Show contours")
        contour_button.clicked.connect(self.show_contour)

        distance_button = QPushButton("Show distance")
        distance_button.clicked.connect(self.show_distance)

        layout = QVBoxLayout()
        layout.addWidget(reset_button)
        layout.addWidget(contour_button)
        layout.addWidget(distance_button)

        self.setLayout(layout)
    
    def show_webcam(self):
        webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = webcam.read()
        try:
            if frame == None:
                webcam = cv2.VideoCapture(-1)
        except:
            pass
        while True:
            ret, frame = webcam.read()
            cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        webcam.release()
        cv2.destroyAllWindows() 

    def show_contour(self):
        webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = webcam.read()
        try:
            if frame == None:
                webcam = cv2.VideoCapture(-1)
        except:
            pass
        while True:
            ret, frame = webcam.read()
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grayscale, (9,9), 0)
            threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3)
            cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 1000:
                    cv2.drawContours(frame, [c], -1, (36, 255, 12), 1)
            cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        webcam.release()
        cv2.destroyAllWindows()
    
    def show_distance(self):
        ARUCO_DICT = {
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        }

        def aruco_display(corners, ids, rejected, frame):
            if len(corners) > 0:
                ids = ids.flatten()
                
                for (markerCorner, markerID) in zip(corners, ids):
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    cv2.line(frame, topLeft, topRight, (255, 0, 255), 2)
                    cv2.line(frame, topRight, bottomRight, (255, 0, 255), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (255, 0, 255), 2)
                    cv2.line(frame, bottomLeft, topLeft, (255, 0, 255), 2)
                    
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                    
                    # Calculate distance
                    marker_size = np.linalg.norm(np.array(topRight) - np.array(topLeft))
                    distance = calculate_distance(marker_size)
                    print("[Inference] ArUco marker ID: {}, Distance: {} units".format(markerID, distance))

                    outlineText = "ID: " + str(markerID) + " at " +  str(round(distance,2)) + " feet"

                    cv2.putText(frame, outlineText,(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)
                    
            return frame


        def calculate_distance(marker_size):
            # Constants for your specific camera setup
            # You need to calibrate these values for your camera
            marker_size_at_one_meter = 0.3332371  # Adjust this value based on the actual marker size at 1 meter distance
            focal_length = 100  # Focal length of your camera in pixels
            
            # Convert meters to feet
            distance_in_feet = marker_size_at_one_meter * focal_length / marker_size * 3.28084
            print(distance_in_feet)
            return distance_in_feet


        aruco_type = "DICT_4X4_100"
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
        arucoParams = cv2.aruco.DetectorParameters_create()

        webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while webcam.isOpened():
            ret, frame = webcam.read()

            h, w, _ = frame.shape
            width = 1000
            height = int(width*(h/w))
            img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

            corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

            frame = aruco_display(corners, ids, rejected, frame)

            cv2.imshow("test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        webcam.release()
