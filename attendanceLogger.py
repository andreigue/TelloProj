import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from djitellopy import tello

me = tello.Tello() #create an object
me.connect() #connect to drone
print(me.get_battery())

me.streamon() #continuous number of frames

# asd = True
intruderNum = 1
path = "Allies"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for person in myList:
    curImg = cv2.imread(f'{path}/{person}')
    images.append(curImg)
    classNames.append(os.path.splitext(person)[0]) #remove ".png" extensions
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open("attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList: #get first time of appearance. If already in the list, then don't add name/time again
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
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
    print("Before seeing a face ========================= ")
    print("faceLocsCurFrame: " + str(faceLocsCurFrame))
    for faceLoc, faceEncode in zip(faceLocsCurFrame, encodeCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, faceEncode, tolerance=0.5)
        faceDist = face_recognition.face_distance(encodeListKnown, faceEncode)
        print(faceDist)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]: #compare_faces() gives a list of [true false] values, depending if threshold is passed
            # case known person or known Intruders
            name = classNames[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1+6, y2+24),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            markAttendance(name)
        else:
            # case new Intruders
            print("Intruders Alert")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            #take snapshot of the Intruders and save in folder
            margin = 60
            y1 = np.clip(y1-margin, 0, img.shape[0])
            y2 = np.clip(y2+margin, 0, img.shape[0])
            x1 = np.clip(x1-margin, 0, img.shape[1])
            x2 = np.clip(x2+margin, 0, img.shape[1])
            intruder_face = img[y1:y2, x1:x2]
            encodingCheck = face_recognition.face_encodings(intruder_face)
            if len(encodingCheck) == 0:
                continue
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            cv2.imwrite(f'Allies/Intruders{intruderNum}.PNG', intruder_face)

            #update known faces
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodeIntruder = face_recognition.face_encodings(intruder_face)[0]
            encodeListKnown.append(encodeIntruder)
            classNames.append(f"Intruders{intruderNum}")
            intruderNum = intruderNum + 1

            # draw red box around Intruders in video
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "INTRUDER", (x1 + 6, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Webcam", img)
    # time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


