"""Drone flies up, rotates around, and when a non-ally face is found, the drone follows the intruder's face.
Once the intruder flees, the drone goes back to rotating."""

import time
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from djitellopy import tello
import faceFollower

me = tello.Tello() #create an object
me.connect() #connect to drone
print(me.get_battery())

me.streamon() #continuous number of frames

me.takeoff()

pError = faceFollower.pError
intruderFound = False
intruderList = []
intruderNum = 1
allypath = "../TelloProj2/Allies"
allyImages = []
allyNames = []
allyList = os.listdir(allypath) #get list of images in Allies/ folder
#
for person in allyList:
    curImg = cv2.imread(f'{allypath}/{person}')
    allyImages.append(curImg)
    allyNames.append(os.path.splitext(person)[0]) #remove ".png" extensions
print(allyNames)

# compute encodings of all images in Allies folder
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListAllies = findEncodings(allyImages)
print(" === Encoding Complete === ")

# cap = cv2.VideoCapture(0)

while True:
    # _, img = cap.read()
    img = me.get_frame_read().frame  # give individual image

    # img = me.get_frame_read().frame #give individual image
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) #quarter of the original size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocsCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceLocsCurFrame)

    while not intruderFound:
        # [lr, fb, ud, yv] : left/right, forward/backwards, up/down, yaw velocity
        me.send_rc_control(0, 0, 0, 50)  # rotate
        time.sleep(1)
        print("New Rotation")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            me.land()
            break

    while intruderFound:
        # _,img = cap.read()
        img = me.get_frame_read().frame  # give individual image
        img = cv2.resize(img, (faceFollower.w, faceFollower.h))
        img, info = faceFollower.findFace(img)
        pError = faceFollower.trackFace(me, info, faceFollower.w, faceFollower.pid,
                           pError)  # the error which trackFace returns will be the previous error for the next iteration
        print("Center :   ", info[0], "   Area :     ", info[1])
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            me.land()
            break

    for faceLoc, faceEncode in zip(faceLocsCurFrame, encodeCurFrame):
        matches = face_recognition.compare_faces(encodeListAllies, faceEncode, tolerance=0.5)
        faceDist = face_recognition.face_distance(encodeListAllies, faceEncode)
        print(faceDist)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]: #compare_faces() gives a list of [true false] values, depending if threshold is passed
            # case known person or known Intruders
            name = allyNames[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1+6, y2+24),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            print(name + " detected")
        else:
            # case new intruder
            print("Intruders Alert")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            #take snapshot of the intruder and save in folder
            margin = 60
            y1 = np.clip(y1-margin, 0, img.shape[0])
            y2 = np.clip(y2+margin, 0, img.shape[0])
            x1 = np.clip(x1-margin, 0, img.shape[1])
            x2 = np.clip(x2+margin, 0, img.shape[1])
            intruder_face = img[y1:y2, x1:x2]
            # encodingCheck = face_recognition.face_encodings(intruder_face)
            # if len(encodingCheck) == 0:
            #     continue
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            cv2.imwrite(f'Intruders{intruderNum}_{dtString}.PNG', intruder_face)
            intruderList.append(f"Intruders{intruderNum}")
            intruderNum = intruderNum + 1
            intruderFound = True

            # draw red box around Intruder in video
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "INTRUDER", (x1 + 6, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera", img)
    # time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


