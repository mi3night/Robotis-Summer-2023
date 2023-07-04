import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QHBoxLayout, QTextEdit, QPlainTextEdit, QMessageBox, QGridLayout, QSizePolicy
import time

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (255, 0, 255), 2)
            cv2.line(image, topRight, bottomRight, (255, 0, 255), 2)
            cv2.line(image, bottomRight, bottomLeft, (255, 0, 255), 2)
            cv2.line(image, bottomLeft, topLeft, (255, 0, 255), 2)
            
            #Object's center pixel coordinates
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            objectCord = "(" + str(cX) + ", " + str(cY) + ")" 
            cv2.putText(image, objectCord, (cX,cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            print("At pixel coordinates ({}, {})".format(cX,cY))


            #Frame center pixel
            h,w,_ = image.shape
            fX=int(w/2)
            fY=int(h/2)
            #Line b/w center and object
            cv2.line(image, (fX,fY), (cX,cY), (255, 0, 0), 2)
            cv2.circle(image, (fX,fY), 3, (255, 0, 0), -1)
            cv2.putText(image," (" + str(fX) + " , " + str(fY) + ")", (fX,fY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            

            # Calculate distance
            marker_size = np.linalg.norm(np.array(topRight) - np.array(topLeft))
            distance = calculate_distance(marker_size)
            print("[Inference] ArUco marker ID: {}, Distance: {} feet\n".format(markerID, distance))
            outlineText = "ID: " + str(markerID) + " at " +  str(round(distance,2)) + " feet"

            cv2.putText(image, outlineText,(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 255), 2)
            
            

            
    return image


def calculate_distance(marker_size):
    # Constants for your specific camera setup
    # You need to calibrate these values for your camera
    marker_size_at_one_meter = 0.3332371  # Adjust this value based on the actual marker size at 1 meter distance
    focal_length = 100  # Focal length of your camera in pixels
    
    # Convert meters to feet
    distance_in_feet = marker_size_at_one_meter * focal_length / marker_size * 3.28084
    return distance_in_feet

class CustomButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)

        # Set the button style
        self.setStyleSheet("""
            QPushButton {
                background-color: Green;
                border: none;
                color: #ffffff;
                border-radius: 10px;
                padding: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #ff3333;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROBOTIS OpenManipulatorX Controller")
        self.setGeometry(100, 100, 800, 500)  # Set the window size to 800x800 pixels

        # Create a QLabel to display the video feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Create QLineEdit widgets for text input
        self.textbox1 = QLineEdit(self)
        self.textbox2 = QLineEdit(self)
        self.textbox3 = QLineEdit(self)

        # Create QLabel widgets for the text box labels
        label1 = QLabel("X:")
        label2 = QLabel("Y:")
        label3 = QLabel("Z:")

        # Set the maximum width for the text boxes
        self.textbox1.setMaximumWidth(200)
        self.textbox2.setMaximumWidth(200)
        self.textbox3.setMaximumWidth(200)

        # Create a Input Coordinate button
        self.coordinate_button = CustomButton("Input Coordinates")
        self.coordinate_button.clicked.connect(self.Input_Coord)
        self.coordinate_button.setMaximumWidth(160)

        # Create an exit button
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: Gray;
                border: none;
                color: #ffffff;
                border-radius: 10px;
                padding: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #ff3333;
            }
            """)
        self.exit_button.setMaximumWidth(160)

        #create show contour button
        contour_toggle = QPushButton("Show Contours")
        contour_toggle.clicked.connect(self.show_contours)

        #create show distance button
        distance_toggle = QPushButton("Show Distance")
        distance_toggle.clicked.connect(self.show_distance)


        #create servo position button
        servo_toggle = QPushButton("Servos Positions")
        servo_toggle.clicked.connect(self.servo_position)

        #ROBOTIS logo

        self.rimage = QPixmap("ROBOTIS.png")
        self.ROBOTIS = QLabel()
        self.ROBOTIS.setPixmap(self.rimage)
        self.ROBOTIS.setAlignment(Qt.AlignBottom)

        # Create a QHBoxLayout for each label and textbox pair
        layout1 = QHBoxLayout()
        layout1.addWidget(label1)
        layout1.addWidget(self.textbox1)
        layout1.addStretch()

        layout2 = QHBoxLayout()
        layout2.addWidget(label2)
        layout2.addWidget(self.textbox2)
        layout2.addStretch()

        layout3 = QHBoxLayout()
        layout3.addWidget(label3)
        layout3.addWidget(self.textbox3)
        layout3.addStretch()

        #output terminal
        label4 = QLabel("Output Terminal:")
        self.output_terminal = QPlainTextEdit()
        self.output_terminal.setReadOnly(True)
        self.output_terminal.setGeometry(QtCore.QRect(10, 90, 331, 111))
        self.output_terminal.setFont(QFont("DejaVu Sans Mono", 8))
        self.output_terminal.setStyleSheet("color: white;"
                        "background-color: black;"
                        "selection-color: black;"
                        "selection-background-color: white;")
        self.output_terminal.setMaximumWidth(600)
        self.coordinate_button.setMaximumHeight(30)
        

        # Create a QVBoxLayout to arrange the widgets
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.video_label)
        middle_layout.addWidget(label4)
        middle_layout.addWidget(self.output_terminal)
        middle_layout.setSpacing(10)
        middle_layout.addStretch()
# -----------------------------kyle additions---------------
        #create QVBoxLayout to add the layout on the left
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.ROBOTIS)

        #create pushbuttons for layout on the right
        '''toggle_servo_position = QPushButton(self)
        toggle_contour = QPushButton(self)
        toggle_distance = QPushButton(self)'''

        #arrow buttons
        #up
        toggle_up = QPushButton(self)
        toggle_up.setMaximumHeight(150)
        toggle_up.setMaximumWidth(150)
        toggle_up.setIcon(QIcon("uparrow.png"))
        #toggle_up.clicked.connect(self.move_up)
        #down
        toggle_down = QPushButton(self)
        toggle_down.setMaximumHeight(150)
        toggle_down.setMaximumWidth(150)
        toggle_down.setIcon(QIcon("downarrow.png"))
        #toggle_down.clicked.connect(self.move_down)
        #right
        toggle_right = QPushButton(self)
        toggle_right.setMaximumHeight(150)
        toggle_right.setMaximumWidth(150)
        toggle_right.setIcon(QIcon("rightarrow.png"))
        #toggle_right.clicked.connect(self.move_right)
        #left
        toggle_left = QPushButton(self)
        toggle_left.setMaximumHeight(150)
        toggle_left.setMaximumWidth(150)
        toggle_left.setIcon(QIcon("leftarrow.png"))
        #toggle_left.clicked.connect(self.move_left)


        button_grid = QGridLayout()
        button_grid.addWidget(toggle_up, 0,1)
        button_grid.addWidget(toggle_down, 3,1)
        button_grid.addWidget(toggle_right, 2,2)
        button_grid.addWidget(toggle_left, 2,0)
        button_grid.setSpacing(2)


    
        #create QVBoxLayout to add the layout on the right
        right_layout = QVBoxLayout()
        right_layout.addWidget(servo_toggle)
        right_layout.addWidget(contour_toggle)
        right_layout.addWidget(distance_toggle)
        right_layout.addLayout(layout1)
        right_layout.addLayout(layout2)
        right_layout.addLayout(layout3)
        right_layout.addWidget(self.coordinate_button)
        right_layout.addLayout(button_grid)
        right_layout.addWidget(self.exit_button, alignment=QtCore.Qt.AlignBottom)
        right_layout.addStretch()
        right_layout.setSpacing(20)

        #create layout combining all the layouts
        total_layout = QHBoxLayout()
        total_layout.addLayout(left_layout)
        total_layout.addLayout(middle_layout)
        total_layout.addLayout(right_layout)
        total_layout.setSpacing(30)
        total_layout.addStretch()

        # Set the main layout for the widget
        self.setLayout(total_layout)

#-----------------------------------------kyle addition-------------------------

        # Set a flag to track whether the Enter key was pressed
        self.enter_pressed = False

        # Open the video source
        self.capture = cv2.VideoCapture(0)

        # Start the video playback
        self.play()

    def play(self):
        self.capture = cv2.VideoCapture(0)
        # Read the next video frame
        ret, frame = self.capture.read()

        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to 800x800 pixels
            frame_resized = cv2.resize(frame_rgb, (600, 450))

            # Create a QImage from the resized frame
            image = QImage(
                frame_resized.data,
                frame_resized.shape[1],
                frame_resized.shape[0],
                QImage.Format_RGB888
            )
            
            corners, ids, rejected = cv2.aruco.detectMarkers(frame_resized, arucoDict, parameters=arucoParams)

            detected_markers = aruco_display(corners, ids, rejected, frame_resized)


            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(image)

            # Set the QPixmap as the image in the QLabel
            self.video_label.setPixmap(pixmap)
            self.video_label.setScaledContents(True)
            # Call the play method again after 15 milliseconds (change the delay as needed)
            QTimer.singleShot(15, self.play)

    def Input_Coord(self):
        # This method will be called when the button is clicked
        # It reads the text from the text boxes
        text1 = self.textbox1.text()
        text2 = self.textbox2.text()
        text3 = self.textbox3.text()

        # Perform any desired actions with the text inputs
        if self.textbox1.text() and self.textbox2.text() and self.textbox3.text():
            print("X:", text1)
            self.output_terminal.appendPlainText("X: ")
            self.output_terminal.insertPlainText(str(text1))
            print("Y:", text2)
            self.output_terminal.appendPlainText("Y: ")
            self.output_terminal.insertPlainText(str(text2))
            print("Z:", text3)
            self.output_terminal.appendPlainText("Z: ")
            self.output_terminal.insertPlainText(str(text3))

            # Clear the text boxes
            self.textbox1.clear()
            self.textbox2.clear()
            self.textbox3.clear()
        else:
            no_input = QMessageBox.critical(self, 'No Input', 'One or more of the coordinates are missing inputs. Please enter a coordin',
            QMessageBox.Retry)

    def closeEvent(self, event):
        # Release the video source when the window is closed
        reply = QMessageBox.question(self, 'Quit', 'Are you sure you want to quit?',
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.capture.isOpened():
                self.capture.release()
            event.accept()
        else:
            event.ignore()
        

    def show_distance(self):
        print("distance")
    
    def show_contours(self):
        print("contour")

    def servo_position(self):
        print("servo position")

aruco_type = "DICT_5X5_100"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters_create()


# Create the QApplication
app = QApplication([])

# Create an instance of the VideoPlayer class
video_player = VideoPlayer()

# Show the window
video_player.show()

# Run the application event loop
app.exec_()
