import pygame

def init():
    pygame.init()
    win = pygame.display.set_mode((400,400))


def getKey(keyName):
    ''' This function will return True if the given keyName has been pressed. False otherwise.'''
    ans = False #if key is pressed, will return True. If key is not pressed, will return False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName)) #if keyName is LEFT, then myKey will be K_LEFT
    if keyInput[myKey]:
        ans = True
    pygame.display.update()

    return ans

def main():
    if getKey("LEFT"):
        print("left key is pressed")
    if getKey("RIGHT"):
        print("right key is pressed")

if __name__ == '__main__':
    init()
    while True:
        main()