import os
import threading as thread

def startpy():
    os.system("python start.py")

def movepy():
    os.system("python move.py")

def lookpy():
    os.system("python look.py")

start = thread.Thread(target=startpy)
move = thread.Thread(target=movepy)
look = thread.Thread(target=lookpy)

start.start()
move.start()
look.start()



