"""Merge code from videoCapture.py and keyBoardControl.py to create a surveillance drone which
will be controllable via keyboard and we can take a picture at any given time."""
from djitellopy import tello
import keyPressModule as kp
import time
import cv2

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

global img

me.streamon() #continuous number of frames

def getKeyboardInput():
    '''Check key presses'''
    lr, fb, ud, yv = 0,0,0,0 #left/right, forward/backwards, up/down, yaw velocity.
    speed = 50

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("d"): yv = speed
    elif kp.getKey("a"): yv = -speed

    if kp.getKey("q"): me.land()
    if kp.getKey("t"): me.takeoff()

    if kp.getKey("f"): me.flip_forward()

    #save image
    if kp.getKey("z"):
        cv2.imwrite(f'Images/{time.time()}.jpg', img)
        time.sleep(0.3)

    return [lr, fb, ud, yv] #return list

while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    img = me.get_frame_read().frame #give individual image
    img = cv2.resize(img, (360,240)) #reduce image to process faster
    cv2.imshow("Video", img)
    time.sleep(0.1)


