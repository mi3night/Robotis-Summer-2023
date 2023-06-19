import cv2
cap = cv2.VideoCapture(0)
# from test5 import *
import numpy as np
while cap.isOpened():
    ret, frame = cap.read()
  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    font = cv2.FONT_HERSHEY_SIMPLEX 
    # detected_markers = aruco_display(corners, ids, rejected, img)
    
    # POSE ESTIMATION
    mtx = [ 5.3434144943931358e+02, 0., 3.3915527161587960e+02, 0.,
       5.3468426373933187e+02, 2.3384359772404599e+02, 0., 0., 1. ]
    mtx = np.array(mtx).reshape(3,3)
    dist = [ -2.8832101897032286e-01, 5.4108203853130338e-02,
       1.7350158737988979e-03, -2.6133116527546341e-04,
       2.0410981186534721e-01 ]
    dist = np.array(dist).reshape(1,5)
    if np.all(ids != None):
        rvec, tvec,_ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.1, mtx, dist) 
        # transformed_rvec = np.asarray(cv2.Rodrigues(rvec))
        # print(transformed_rvec)
        gray = cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.1)  # Draw Axis
    
        
    cv2.imshow("Image", gray)
    key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()