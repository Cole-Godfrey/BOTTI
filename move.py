import time, interception, random
interception.auto_capture_devices()

def forward():
    interception.key_down("w")
    time.sleep(1)
    interception.key_up("w")


def left():
    interception.key_down("a")
    time.sleep(1)
    interception.key_up("a")


def right():
    interception.key_down("d")
    time.sleep(1)
    interception.key_up("d")


def backward():
    interception.key_down("s")
    time.sleep(1)
    interception.key_up("s")


def forwardright():
    interception.key_down("w")
    interception.key_down("d")
    time.sleep(1)
    interception.key_up("w")
    interception.key_up("d")


def forwardleft():
    interception.key_down("w")
    interception.key_down("a")
    time.sleep(1)
    interception.key_up("w")
    interception.key_up("a")


def backwardright():
    interception.key_down("s")
    interception.key_down("d")
    time.sleep(1)
    interception.key_up("s")
    interception.key_up("d")


def backwardleft():
    interception.key_down("s")
    interception.key_down("a")
    time.sleep(1)
    interception.key_up("s")
    interception.key_up("a")


def jump():
    interception.key_down("l")
    time.sleep(1)
    interception.key_up("l")

def c():
    interception.key_down("c")
    interception.key_up("c")
    time.sleep(1)
    interception.left_click()

def q():
    interception.key_down("q")
    interception.key_up("q")
    time.sleep(1)
    randy = random.randint(0,1)
    if randy == 0:
        interception.left_click()
    else:
        interception.right_click()


def e():
    interception.key_down("e")
    interception.key_up("e")
    time.sleep(1)
    randy = random.randint(0,1)
    if randy == 0:
        interception.left_click()
    else:
        interception.right_click()

def x():
    interception.key_down("x")
    interception.key_up("x")

def slowforward():
    interception.key_down("p")
    interception.key_down("w")
    time.sleep(1)
    interception.key_up("p")
    interception.key_up("w")


options = {
    0: forward,
    9: forward,
    10: forward,
    11: forward,
    12: forward,
    1: backward,
    2: left,
    3: right,
    4: forwardleft,
    5: forwardright,
    6: backwardleft,
    7: backwardright,
    8: jump,
    13: c,
    14: slowforward,
    15: q,
    16: slowforward,
    17: e,
    18: slowforward,
    19: x,
    20: slowforward


}
def start():
    time.sleep(10)
    while True:
        rand = random.randint(0, 20)
        options[rand]()

start()