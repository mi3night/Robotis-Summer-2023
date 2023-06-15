import cv2
import cv2.aruco as aruco
import numpy as np

# Load the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Create the ArUco parameters
aruco_params = aruco.DetectorParameters()

# Load the camera calibration parameters (you need to provide your own calibration file)
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers in the grayscale image
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    # If any markers are detected
    if ids is not None:
        # Estimate the pose of the markers
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        
        # Draw the detected markers and their 3D axis
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
            
        # Estimate the distance to the markers
        for i in range(len(ids)):
            distance = np.linalg.norm(tvecs[i])
            print(f"Marker {ids[i][0]}: Distance = {distance} meters")
    
    # Display the resulting frame
    cv2.imshow('AR Marker Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
