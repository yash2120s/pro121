import cv2
import time
import numpy as np


fourcc = cv2.VideoWriter_fourcc(*"XVID")

output_file = cv2.VideoWriter("output100.avi",fourcc,20.0,(640,480))

cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(60):
    ret,bg = cap.read()

#flipping the background
bg = np.flip(bg,axis=1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break 
    #flipping the image for constitency
    img = np.flip(img,axis = 1)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#generating mask to detect red color 
    
    mask = cv2.inRange(hsv,(0,0,0),(140,255,255))
    
    

    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
   
    mask1 = cv2.bitwise_not(mask)

  
    res1 = cv2.bitwise_and(bg,bg,mask = mask)
    res2 = cv2.bitwise_and(img,img,mask = mask1)

    final_output = cv2.addWeighted(res1,1,res2,1,0)
    
    
    output_file.write(final_output)

    cv2.imshow('black screen magic',final_output)
    cv2.waitKey(1)

cap.release()
output_file.release()
cv2.destroyAllWindows()