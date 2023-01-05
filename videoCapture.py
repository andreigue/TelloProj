from djitellopy import tello
import cv2
from time import sleep

me = tello.Tello() #create an object
me.connect() #connect to drone
print(me.get_battery())

me.streamon() #continuous number of frames

while True:
    img = me.get_frame_read().frame #give individual image
    img = cv2.resize(img, (360,240)) #reduce image to process faster
    cv2.imshow("Video", img)
    cv2.waitKey(1) #frame will shut down before we can see it, so give delay of 1ms

