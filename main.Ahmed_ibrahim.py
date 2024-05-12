#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
from simple_facerec import SimpleFacerec
import os
os.chdir(r"C:\ML\computer_vision\Ahmed_ibrahim\face-recognition-arabic-main\imge")
# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("imge1/")

# Load Camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




