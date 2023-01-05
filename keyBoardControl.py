from djitellopy import tello
import keyPressModule as kp
import time

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())


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

    return [lr, fb, ud, yv] #return list

while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    time.sleep(0.1)
