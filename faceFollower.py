import cv2
import numpy as np
from djitellopy import tello




w, h = 360, 240
fbRange = [6200, 6800] #forward/backward range (bounding box area)
heightRange = [110, 130]
pid = [0.4, 0.4, 0] #proportional integral derivative
pError = 0

def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListCenter = []
    myFaceListArea = []

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,0,255),3) #draw bounding box around face
        cx = x + w//2 #center x pixel (rounded down)
        cy = y + h//2
        area = w * h
        cv2.circle(img, (cx,cy),5, (0,255,0), cv2.FILLED)
        myFaceListCenter.append([cx,cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListCenter[i], myFaceListArea[i]]
    else:
        return img, [(0,0), 0]


def trackFace(me, info, w, pid, pError): #pError = previous error
    area = info[1]
    x, y = info [0]
    fb = 0
    ud = 0
    error = x - w//2 # difference between the face center, and the center of the screen (to rotate left or right)
    speed = pid[0] * error + pid[1] * (error-pError)
    speed = int(np.clip(speed, -100, 100)) #values smaller than -100 will be capped at -100, and similar for values over 100

    #compare current area with fbRange
    if area > fbRange[0] and area < fbRange[1]: #in range, no movement
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0

    # modify height
    if y > heightRange[0] and y < heightRange[1]: #in range, no movement
        ud = 0
    elif y > heightRange[1]:
        ud = -20
        print("###################### Going DOWN #######################")
    elif y < heightRange[0] and y != 0 :
        ud = 20
        print("###################### Going UP #######################")

    me.send_rc_control(0, fb, ud, speed)
    return error

# cap = cv2.VideoCapture(0) #capture stream from main camera on laptop

if __name__ == "__main__":

    me = tello.Tello()  # create an object
    me.connect()  # connect to drone
    print(me.get_battery())

    me.streamon()  # continuous number of frames

    me.takeoff()

    while True:
        # _,img = cap.read()
        img = me.get_frame_read().frame #give individual image
        img = cv2.resize(img, (w,h))
        img, info = findFace(img)
        pError = trackFace(me, info, w, pid, pError) #the error which trackFace returns will be the previous error for the next iteration
        print("Center :   ", info[0], "   Area :     ", info[1])
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            me.land()
            break