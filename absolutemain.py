import os
import threading as thread

def startpy():
    os.system("python start.py")

def movementpy():
    os.system("python movement.py")

start = thread.Thread(target=startpy)
move = thread.Thread(target=movementpy)


start.start()
move.start()



