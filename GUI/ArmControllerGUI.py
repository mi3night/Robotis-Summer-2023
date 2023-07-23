import cv2
import numpy as np
import os
import motorctrl_v1 as motor
import Movement_Calc_v2 as calculation
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon,QKeySequence, QKeyEvent
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QHBoxLayout, QPlainTextEdit, QMessageBox, QGridLayout, QSizePolicy

#Filepath for images and obj_detect setup
os.chdir(r'icons')

#Toggle
AR_flag = 0
Obj_flag = 0
mode = 0
#Arm Setup
BASE_ID = 1
BICEP_ID = 2
FOREARM_ID = 3
WRIST_ID = 4
CLAW_ID = 0

PORT_NUM = 'COM5'
BAUDRATE = 1000000

MOVEARM_MODE = 1

ALL_IDs = [BASE_ID, BICEP_ID, FOREARM_ID, WRIST_ID, CLAW_ID]
MOVE_IDs = [BASE_ID, BICEP_ID, FOREARM_ID, WRIST_ID]

frameX = 0
objX = 0
frameY = 0
objY = 0

x = 20
y = 0
z = -150

motor.portInitialization(PORT_NUM, ALL_IDs)
motor.dxlSetVelo([20, 20, 20, 20, 20], [0, 1, 2, 3, 4])
motor.motorRunWithInputs([228], [4])
motor.motorRunWithInputs([90, 227, 273, 47], [0, 1, 2, 3])

#Obj Detect setup
classesFile = r'coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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

#AR marker setup
aruco_type = "DICT_5X5_100"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters_create()

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
            h,w,_ = image.shape
            global objX
            global frameX
            global objY
            global frameY
            fX=int(w/2)
            frameX = fX
            fY=int(h/2)
            frameY = fY
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            objX = cX
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            objY = cY
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            objectCord = "(" + str(cX) + ", " + str(cY) + ")" 
            cv2.putText(image, objectCord, (cX,cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            print("At pixel coordinates ({}, {})".format(cX,cY))

            # Calculate distance
            
            marker_size = np.linalg.norm(np.array(topRight) - np.array(topLeft))
            distance_feet, distance_per_pixel = calculate_distance(marker_size)
            distance_feet_rounded = round(distance_feet, 2)
            distance_per_pixel_rounded = round(distance_per_pixel, 6)
            arX = distance_per_pixel * (cX - fX) / 3.6
            arY = distance_feet_rounded
            arX = round(arX, 4)
            arY = round(arY, 4)
            print("[Inference] ArUco marker ID: {}, Distance: {} feet, Distance per pixel: {} feet/pixel\n, X coord: {}, Y coord: {}\n".format(markerID, distance_feet_rounded, distance_per_pixel_rounded, arX, arY))
            outlineText = "ID: " + str(markerID) + " at " +  str(distance_feet_rounded) + " feet, " +  str(distance_per_pixel_rounded) + " ft/pixel" 
            outlineText2 = "X Axis: " + str(arX) + " Y Axis: " + str(arY)   
            

            cv2.putText(image, outlineText,(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 255), 2)
            cv2.putText(image, outlineText2, (topLeft[0], topLeft[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return image

def findObjects(outputs,img):
    confThreshold = 0.5
    nmsThreshold = 0.3
    hT, wT,_ = img.shape
    bbox = []
    classIds = []
    confs = []
    centerPoints = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                w,h = int(det[2] * wT), int(det[3] * hT)
                x,y = int((det[0] * wT) - w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                centerPoints.append((center_x, center_y))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        center_x, center_y = centerPoints[i]
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
        text = f'({center_x}, {center_y})'
        cv2.putText(img, text, (center_x - 20, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def calculate_distance(marker_size):
    # Constants for your specific camera setup
    # You need to calibrate these values for your camera
    marker_size_at_one_meter = 0.1055  # Adjust this value based on the actual marker size at 1 meter distance
    focal_length = 226  # Focal length of your camera in pixels
    
    # Convert meters to feet
    distance_in_feet = marker_size_at_one_meter * focal_length / marker_size * 3.28084
    distance_per_pixel = distance_in_feet / focal_length

    return distance_in_feet, distance_per_pixel

class ToggleButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {            
                background-color: #DDDDDD;
                color: #000000;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border: 2px solid #555555;
                border-radius: 10px;
            }
            QPushButton:pressed, QPushButton:checked {
                background-color: green;
                border: 2px solid #555555;
            }
        """)

        self.setCheckable(True)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Right, Qt.Key_Left):
            self.setChecked(True)
            print()
            self.setStyleSheet("""
                QPushButton {             
                    background-color: green;
                    color: #000000;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px;
                    border: 2px solid #555555;
                    border-radius: 10px;
                }
            """)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Right, Qt.Key_Left):
            self.setChecked(False)
            self.setStyleSheet("""
                QPushButton {             
                    background-color: #DDDDDD;
                    color: #000000;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px;
                    border: 2px solid #555555;
                    border-radius: 10px;
                }
                QPushButton:pressed {
                    background-color: green;
                    border: 2px solid #555555;
                }
            """)


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

def ArrowMov(direction):
    current = motor._map(motor.ReadMotorData(1, 132), 0, 4095, 0, 360)
    current2 = motor._map(motor.ReadMotorData(2, 132), 0, 4095, 0, 360)
    current3 = motor._map(motor.ReadMotorData(3, 132), 0, 4095, 0, 360)
    current4 = motor._map(motor.ReadMotorData(4, 132), 0, 4095, 0, 360)
    if direction == 0:
        print("UP")
        motor.motorRunWithInputs([(current2 - 6.9), (current3 + 18), (current4 - 7)], [2, 3, 4])
    elif direction == 1:
        print("RIGHT")
        motor.motorRunWithInputs([current - 10], [1])
    elif direction == 2:
        print("DOWN")
        motor.motorRunWithInputs([(current2 + 6.9), (current3 - 18), (current4 + 10)], [2, 3, 4])    
    elif direction == 3:
        print("LEFT")
        motor.motorRunWithInputs([current + 10], [1])    
    else:
        print("Invalid direction:", direction)

class ControllerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROBOTIS OpenManipulatorX Controller")
        self.setGeometry(50, 50, 1000, 800)  # Set the window size to 800x800 pixels

        # Create a QLabel to display the video feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Create QLineEdit widgets for text input
        self.textbox1 = QLineEdit(self)
        self.textbox2 = QLineEdit(self)
        self.textbox3 = QLineEdit(self)

        # Set the maximum width for the text boxes
        self.textbox1.setFixedWidth(100)
        self.textbox2.setFixedWidth(100)
        self.textbox3.setFixedWidth(100)

        # Create QLabel widgets for the text box labels
        label1 = QLabel("X:")
        label2 = QLabel("Y:")
        label3 = QLabel("Z:")

        #ROBOTIS logo
        self.rimage = QPixmap("ROBOTIS.png")
        self.ROBOTIS = QLabel()
        self.ROBOTIS.setPixmap(self.rimage)
        self.ROBOTIS.setAlignment(Qt.AlignBottom)

        # Create a Method buttons
        self.XYZ_button = CustomButton("Input Coordinates")
        self.XYZ_button.setFixedWidth(200)
        self.XYZ_button.clicked.connect(self.Input_Coord)
        

        self.Stop_button = CustomButton("Force stop")
        self.Stop_button.setFixedWidth(200)
        #self.button.clicked.connect(self.Input_Coord)

        self.R_button = CustomButton("Reset")
        self.R_button.setFixedWidth(200)
        self.R_button.clicked.connect(self.ResetPos)
        
        self.activity_status = QPushButton(self)
        self.activity_status.setStyleSheet("border: 1px solid black;")
        self.activity_status.setText(state)

        self.Disp_button = CustomButton("Display")
        self.Disp_button.setFixedWidth(200)
        self.Disp_button.clicked.connect(self.Display)

        self.Mode_button = CustomButton("Mode")
        self.Mode_button.setFixedWidth(200)
        self.Mode_button.clicked.connect(self.Mode)
    
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setFixedWidth(200)
        self.exit_button.clicked.connect(self.close)

        #arrow buttons
        #up
        toggle_up = ToggleButton("",self)
        toggle_up.setFixedWidth(50)
        toggle_up.setFixedWidth(50)
        toggle_up.setIcon(QIcon("uparrow.png"))
        up = QKeySequence(Qt.Key_Up)
        toggle_up.setShortcut(up)
        toggle_up.clicked.connect(lambda: ArrowMov(0))
        #down
        toggle_down = ToggleButton("",self)
        toggle_down.setFixedWidth(50)
        toggle_down.setFixedWidth(50)
        toggle_down.setIcon(QIcon("downarrow.png"))
        down = QKeySequence(Qt.Key_Down)
        toggle_down.setShortcut(down)
        toggle_down.clicked.connect(lambda: ArrowMov(2))
        #right
        toggle_right = ToggleButton("",self)
        toggle_right.setFixedWidth(50)
        toggle_right.setFixedWidth(50)
        right = QKeySequence(Qt.Key_Right)
        toggle_right.setShortcut(right)
        toggle_right.setIcon(QIcon("rightarrow.png"))
        toggle_right.clicked.connect(lambda: ArrowMov(1))
        #left
        toggle_left = ToggleButton("",self)
        toggle_left.setFixedWidth(50)
        toggle_left.setFixedWidth(50)
        toggle_left.setIcon(QIcon("leftarrow.png"))
        left = QKeySequence(Qt.Key_Left)
        toggle_left.setShortcut(left)
        toggle_left.clicked.connect(lambda: ArrowMov(3))

        # Set focus policy to capture arrow keys
        toggle_up.setFocusPolicy(Qt.StrongFocus)
        toggle_down.setFocusPolicy(Qt.StrongFocus)
        toggle_left.setFocusPolicy(Qt.StrongFocus)
        toggle_right.setFocusPolicy(Qt.StrongFocus)

        button_grid = QGridLayout()
        button_grid.addWidget(toggle_up, 0,1)
        button_grid.addWidget(toggle_down, 3,1)
        button_grid.addWidget(toggle_right, 2,2)
        button_grid.addWidget(toggle_left, 2,0)
        button_grid.setSpacing(2)


        # Create a QHBoxLayout for each label and textbox pair
        layout1 = QHBoxLayout()
        layout1.setSpacing(5)
        layout1.addWidget(label1)
        layout1.addWidget(self.textbox1)
        layout1.setAlignment(label1, Qt.AlignRight)  # Align label1 to the right

        layout2 = QHBoxLayout()
        layout2.setSpacing(5)
        layout2.addWidget(label2)
        layout2.addWidget(self.textbox2)
        layout2.setAlignment(label2, Qt.AlignRight)  # Align label2 to the right

        layout3 = QHBoxLayout()
        layout3.setSpacing(5)
        layout3.addWidget(label3)
        layout3.addWidget(self.textbox3)
        layout3.setAlignment(label3, Qt.AlignRight)  # Align label3 to the right

        # Output terminal
        label4 = QLabel("Output Terminal:")
        self.output_terminal = QPlainTextEdit()
        self.output_terminal.setReadOnly(True)
        self.output_terminal.setGeometry(QRect(10, 90, 331, 80))
        self.output_terminal.setFont(QFont("DejaVu Sans Mono", 8))
        self.output_terminal.setStyleSheet("color: white;"
                                            "background-color: black;"
                                            "selection-color: black;"
                                            "selection-background-color: white;")
        self.output_terminal.setMaximumWidth(800)

        # Right side Vertical Layout
        R_v_layout = QVBoxLayout()
        R_v_layout.addStretch()
        R_v_layout.setSpacing(10)  # Set the spacing between items to 10 pixels
        R_v_layout.addWidget(self.Disp_button)
        R_v_layout.addWidget(self.Mode_button)
        R_v_layout.addLayout(layout1)
        R_v_layout.addLayout(layout2)
        R_v_layout.addLayout(layout3)
        R_v_layout.addWidget(self.XYZ_button)
        R_v_layout.addLayout(button_grid)
        R_v_layout.addWidget(self.exit_button)

        # Left side Vertical Layout
        L_v_layout = QVBoxLayout()
        L_v_layout.addStretch()
        L_v_layout.setSpacing(10)
        L_v_layout.addWidget(self.Stop_button)
        L_v_layout.addWidget(self.R_button)
        L_v_layout.addWidget(self.activity_status)
        L_v_layout.addWidget(self.ROBOTIS)

        # Middle layout
        mid_layout = QVBoxLayout()
        mid_layout.setSpacing(10)  # Set the spacing between items to 10 pixels
        mid_layout.addWidget(self.video_label)
        mid_layout.addWidget(label4)
        mid_layout.addWidget(self.output_terminal)

        # Assemble all vertical layouts
        final_layout = QHBoxLayout()
        final_layout.setSpacing(10)

        # Set stretch factors for left and right layouts
        final_layout.addLayout(L_v_layout)
        final_layout.addLayout(mid_layout)
        final_layout.addLayout(R_v_layout)

        # Align the final layout to the left
        final_layout.setAlignment(Qt.AlignLeft)

        # Set the main layout for the widget
        self.setLayout(final_layout)

        # Open the video source
        self.capture = cv2.VideoCapture(0)

        # Start the video playback
        self.play()

    def play(self):
        whT = 320
        # Read the next video frame
        ret, frame = self.capture.read()

        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a QImage from the resized frame
            image = QImage(
                frame_rgb.data, #RGB image values
                frame_rgb.shape[1], #Width
                frame_rgb.shape[0], #Height
                QImage.Format_RGB888
            )
            #Center frame pixel
            fX=int(frame_rgb.shape[1]/2)
            fY=int(frame_rgb.shape[0]/2)
            cv2.circle(frame_rgb, (fX,fY), 3, (255, 0, 0), -1)
            cv2.putText(frame_rgb," (" + str(fX) + " , " + str(fY) + ")", (fX,fY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
           

            #Object Detect
            if Obj_flag ==1:
                blob = cv2.dnn.blobFromImage(frame_rgb, 1/255, (whT,whT), [0,0,0], 1, crop=False)
                net.setInput(blob)
                layerNames = net.getLayerNames()
                outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                findObjects(outputs, frame_rgb)
            #Aruco detect
            if AR_flag == 1:
                corners, ids, rejected = cv2.aruco.detectMarkers(frame_rgb, arucoDict, parameters=arucoParams)
                aruco_display(corners, ids, rejected, frame_rgb)

            #Testing team: AR marker tracking
            if mode == 1:
                if(abs(objX - frameX) > 30):
                    difference = objX - frameX
                    print('x difference: ' + str(difference))
                    current = motor._map(motor.ReadMotorData(1, 132), 0, 4095, 0, 360)
                    print("current: " + str(current))
                    if (difference < 10 and ids is not None):
                        pass
                        motor.WriteMotorData(1, 116, current - 10)
                        motor.motor_check(1,motor._map(current - 10 , 0, 360, 0, 4095))
                        motor.dxlSetVelo([37],[1])
                        motor.motorRunWithInputs([current - difference/20], [1])
                    elif (difference > 10 and ids is not None):
                        pass
                        motor.dxlSetVelo([37],[1])
                        motor.motorRunWithInputs([current - difference/20], [1])

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
        x_inp = self.textbox1.text()
        y_inp = self.textbox2.text()
        z_inp = self.textbox3.text()
        x_move = 0
        y_move = 0
        z_move = 0

        # Output terminal XYZ input
        if self.textbox1.text() and self.textbox2.text() and self.textbox3.text():
            motor.portInitialization(PORT_NUM, ALL_IDs)
            x_move = int(x_inp)
            y_move = int(y_inp)
            z_move = int(z_inp)
        #From robotic arm code
            coor = [x_move,y_move,z_move]
            angles = calculation.angle_Calc(coor, 0)
            print(angles)
            motor.dxlSetVelo([30,18,30,30,30], ALL_IDs)
            motor.simMotorRun(angles, MOVE_IDs)

            # Clear the text boxes
            self.textbox1.clear()
            self.textbox2.clear()
            self.textbox3.clear()
            self.output_terminal.appendPlainText("X: ")
            self.output_terminal.insertPlainText(str(x_inp))
            self.output_terminal.appendPlainText("Y: ")
            self.output_terminal.insertPlainText(str(y_inp))
            self.output_terminal.appendPlainText("Z: ")
            self.output_terminal.insertPlainText(str(z_inp))
        else:
            no_input = QMessageBox.critical(self, 'No Input', 'One or more of the coordinates are missing inputs. Please enter a coordin',
            QMessageBox.Retry)

        def statuscheck():

            global state
            state = None

            if motor.motor_status == 0:
                state = 'stationary..'
                return (state)
    
            elif motor.motor_status == 1:
                state = 'running..'
                return (state)
    
            else:
                state = 'error'
                return(state)
            
        statuscheck()
    
    def Display(self):
        # Access flags
        global AR_flag
        global Obj_flag

        # Create a QMessageBox
        message_box = QMessageBox()
        message_box.setWindowTitle("Display Toggle")
        message_box.setText("Choose an option:")

        # Add buttons in the desired order
        ar_marker_button = message_box.addButton("AR Marker", QMessageBox.AcceptRole)
        obj_detect_button = message_box.addButton("Object Detect", QMessageBox.DestructiveRole)
        cancel_button = message_box.addButton("Cancel", QMessageBox.RejectRole)

        # Execute the message box
        message_box.exec_()

        # Get the role of the clicked button
        clicked_button_role = message_box.buttonRole(message_box.clickedButton())

        # Handle the clicked button
        if clicked_button_role == QMessageBox.AcceptRole:
            # Handle AR Marker option
            if AR_flag == 1:
                self.output_terminal.appendPlainText("AR Marker: OFF")
                AR_flag = 0
            else:
                self.output_terminal.appendPlainText("AR Marker: ON")
                AR_flag = 1
        elif clicked_button_role == QMessageBox.DestructiveRole:
            # Handle Object Detect option
            if Obj_flag == 1:
                self.output_terminal.appendPlainText("Object Detect: OFF")
                Obj_flag = 0
            else:
                self.output_terminal.appendPlainText("Object Detect: ON")
                Obj_flag = 1
        elif clicked_button_role == QMessageBox.RejectRole:
            # Handle Cancel option
            pass
        else:
            self.output_terminal.appendPlainText("Unknown button clicked")
    
    def Mode(self):
            global mode
            global AR_flag
            # Create a QMessageBox
            message_box = QMessageBox()
            message_box.setWindowTitle("Toggle Arm mode")
            message_box.setText("Choose an option:")

            # Add buttons in the desired order
            tracking_button = message_box.addButton("AR marker tracking", QMessageBox.AcceptRole)
            TBD_button = message_box.addButton("TBD", QMessageBox.DestructiveRole)
            cancel_button = message_box.addButton("Cancel", QMessageBox.RejectRole)

            # Execute the message box
            message_box.exec_()

            # Get the role of the clicked button
            clicked_button_role = message_box.buttonRole(message_box.clickedButton())

            # Handle the clicked button
            if clicked_button_role == QMessageBox.AcceptRole:
                # Handle AR marker tracking
                if mode == 1:
                    self.output_terminal.appendPlainText("AR marker tracking: OFF")
                    mode = 0
                else:
                    self.output_terminal.appendPlainText("AR Marker tracking: ON")
                    if AR_flag == 0:
                        AR_flag = 1
                        self.output_terminal.appendPlainText("AR marker: ON")
                    mode = 1
            elif clicked_button_role == QMessageBox.DestructiveRole:
                # Handle Object Detect option
                if mode == 2:
                    self.output_terminal.appendPlainText("Nothing happened")
                    mode = 0
                else:
                    self.output_terminal.appendPlainText(".")
                    mode = 2
            elif clicked_button_role == QMessageBox.RejectRole:
                # Handle Cancel option
                pass
            else:
                self.output_terminal.appendPlainText("Unknown button clicked")
    
    def ResetPos(self):
        self.output_terminal.appendPlainText("Moving to default position")
        motor.dxlSetVelo([20, 20, 20, 20, 20], [0, 1, 2, 3, 4])
        motor.motorRunWithInputs([227], [4])
        motor.motorRunWithInputs([90, 227, 273, 47], [0, 1, 2, 3])

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


def main():
    # Create the QApplication
    app = QApplication([])

    # Create an instance of the VideoPlayer class
    GUI = ControllerGUI()

    # Show the window
    GUI.show()

    # Run the application event loop
    app.exec_()

if __name__=="__main__":
    main()
