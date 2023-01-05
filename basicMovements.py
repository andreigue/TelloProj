from djitellopy import tello
from time import sleep #will need delays between commands

me = tello.Tello() #create a tello object
me.connect() #connect to drone (takes care of all the IP address/communication part)

print(me.get_battery())

me.takeoff()
sleep(2)
me.send_rc_control(0,40,0,0) #move with % velocity. If want distance to travel, use move_XYZ() function
sleep(2) #moves at the speed of 40 for 2 seconds
me.send_rc_control(0,0,0,0)
sleep(1)
me.send_rc_control(0,0,0, 60) #rotate 90 degrees to the right
sleep(2)
me.send_rc_control(40,0,0,0)
sleep(2)
me.send_rc_control(0,0,0,0) #stop the drone before landing
me.land()