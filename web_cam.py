
import ultralytics                   
from ultralytics import YOLO         
import cv2                           
import cvzone                       
import math                         
import numpy as np                  
from sort import *                   


cap = cv2.VideoCapture('2103099-uhd_3840_2160_30fps.mp4')


cap.set(3, 255)
cap.set(4, 255)

model = YOLO("yolov8n.pt")


mask = cv2.imread('m.png')
mask = cv2.resize(mask, (1280, 720))


line = [140, 600, 1182, 600]  # [x1, y1, x2, y2]


total_count = []


class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "Mobile", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)


while True:

    ret, frame = cap.read()


    if not ret:
        break


    frame = cv2.resize(frame, (1280, 720))


    imageRegion = cv2.bitwise_and(frame, mask)


    results = model(imageRegion, verbose=False)


    stop_detecting = False


    detections = np.empty((0, 5))

    
    for r in results:
        boxes = r.boxes

        
        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

           
            w, h = x2 - x1, y2 - y1

           
            conf = math.ceil((box.conf * 100)) / 100

            
            cls = int(box.cls[0])

            
            if cls == 67:
                stop_detecting = True
                break

            
            if class_names[cls] in ['car', 'bus', 'truck']:
                
                cvzone.cornerRect(frame, (x1, y1, w, h), l=30, t=5, colorR=(0, 0, 0), colorC=(200, 200, 200))

               
                currentArray = np.array([x1, y1, x2, y2, conf])

               
                detections = np.vstack((detections, currentArray))

   
    if stop_detecting:
        break

   
    resulttracker = tracker.update(detections)

    
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 5)

    
    for result in resulttracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        w, h = x2 - x1, y2 - y1

        
        cvzone.cornerRect(frame, (x1, y1, w, h), l=30, t=5, colorR=(0, 150, 0), colorC=(200, 200, 200))

        
        cvzone.putTextRect(frame, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, colorT=(0, 0, 0), colorB=(255, 255, 255), colorR=(255, 255, 255))

        
        cx, cy = x1 + w // 2, y1 + h // 2

       
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), 2)

        
        if line[0] < cx < line[2] and line[1] - 20 < cy < line[3] + 30:
            
            if total_count.count(Id) == 0:
                total_count.append(Id)

        
        cvzone.putTextRect(frame, f'Total Count: {len(total_count)}', (50, 50))

    
    cv2.imshow("Frame", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
