from ultralytics import YOLO
import numpy

model = YOLO("yolov8n.pt", "v8")

detection_output = model.predict(source = "opencv-test/TestImage/img0.JPG", conf = 0.25, save = False)

print(detection_output)

print(detection_output[0].numpy())
