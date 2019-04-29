import cv2
import numpy as np
import torch

state = torch.load('tmp.pth')
print(state.keys())
print(type(state))



cap = cv2.VideoCapture('2_kisi_A.mp4')
delay = 0

while(cap.isOpened()):
    ret, frame = cap.read()


    if delay % 300 == 0 :
        result =state(cap)

        print(result)

        delay +=1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
