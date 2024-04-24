import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to perform object detection on the selected video
def perform_object_detection(video_path):
    # Load YOLO
    net =  cv2.dnn.readNet('cfg/yolov3.weights', 'cfg/yolov3.cfg')
    classes = []
    with open('cfg/coco.names', 'r') as f:
        classes = f.read().splitlines()

    # Load input video
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape

        # Preprocess input image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Forward pass through the network
        output_layers_names = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(output_layers_names)

        # Process detection results
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression to remove redundant overlapping boxes
        indexes
