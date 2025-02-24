from ultralytics import YOLO
from IPython import display
from IPython.display import display, Image
import cv2 as cv
import numpy as np
import matplotlib as plt
import torch
import pandas as pd
import os

Model = YOLO(r"C:\Users\pashi\Krackhack\best.pt")
#Model.train(data = "C:\\Users\\pashi\\Krackhack\\vehical\\configurefile(vehicle).yaml", epochs = 2, imgsz = 640)
#print(Model)
torch.save(Model.state_dict(), "C:\\Users\\pashi\\Krackhack\\vehical\\Vehicle_Detection_Image_Dataset\\model.pt")
# video_path = #enter the link to the video 
# cap = cv.VideoCapture(video_path)
# if not cap.isOpened():
#     print("error in vid")
#     exit()
# total_car_count = 0
# total_person_count = 0
# prev_time = 0
# prev_position = (0, 0)
# while cap.isOpened():
#     ret,frame = cap.read()
#     frame = cv.resize(frame,(1000,800))
#     target_class_name = 'car'
#     target_class_name_1 = 'person'
#     class_names = ['car','bus','motor','bike','train','person','rider','traffic sign']
#     target_class_id = class_names.index(target_class_name)  
#     target_class_id_1 = class_names.index(target_class_name_1)  
#     annotated_frame = Model.track(frame,persist=True)
#     boxes = annotated_frame[0].boxes.xyxy
#     class_ids = annotated_frame[0].boxes.cls
#     scores = annotated_frame[0].boxes.conf
#     car_count = 0
#     person_count = 0
#     person_speed = []
#     if not ret:
#         break
#     for i in range(len(boxes)):
#         if class_ids[i] == target_class_id: 
#            car_count += 1
#         if class_ids[i] == target_class_id_1+1:
#            person_count+=1
#         x1, y1, x2, y2 = boxes[i]
#         center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
#         current_time = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
#         distance = np.linalg.norm((np.array(center)) - np.array(prev_position))
#         time_diff = current_time - prev_time
#         speed = distance/time_diff 
#         #if person_count>0:
#         if  class_ids[i] == target_class_id_1+1:
#             person_speed.append(speed)
#         if len(person_speed)>0:
#             person_speed_avg = sum(person_speed)/len(person_speed)
#         else:
#             person_speed_avg = 0
#         print(speed)
#         if speed>person_speed_avg*2 and time_diff>0:
#             cv.rectangle(frame, (int(x1-10), int(y1-10)), (int(x2+10), int(y2+10)), (0,255,255), 2)

        
            
#     cv.putText(frame, f'Cars Count: {car_count}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
#     cv.putText(frame, f'person Count: {person_count}', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
#     cv.putText(frame,"if speed is twice faster than average persons speed in frame",(10,130),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,234),1,cv.LINE_AA)
#     cv.rectangle(frame,(530,90),(590,150),(0,255,255),thickness=cv.FILLED)
       
    

    
    
#     frame_ = (annotated_frame[0].plot())
    

#     #cv.resizeWindow("YOLO object detection",800,600)
#     cv.imshow("YOLO object detection",frame_)

#     #to display class count

#     if cv.waitKey(1)& 0xFF == ord('q'):
#         break
    

# cap.release()
# cv.destroyAllWindows()
