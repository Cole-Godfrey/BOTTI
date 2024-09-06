import time
import interception
import random

# Set up interception
interception.auto_capture_devices()

# Constants
FRAME_DURATION_SECONDS = 1

# Define movement functions
def move_forward():
    interception.key_down("w")
    time.sleep(FRAME_DURATION_SECONDS)
    interception.key_up("w")
def move_left():
    interception.key_down("a")
    time.sleep(1)
    interception.key_up("a")
def move_right():
    interception.key_down("d")
    time.sleep(1)
    interception.key_up("d")
def move_backward():
    interception.key_down("s")
    time.sleep(1)
    interception.key_up("s")
def move_forward_right():
    interception.key_down("w")
    interception.key_down("d")
    time.sleep(1)
    interception.key_up("w")
    interception.key_up("d")
def move_forward_left():
    interception.key_down("w")
    interception.key_down("a")
    time.sleep(1)
    interception.key_up("w")
    interception.key_up("a")
def move_backward_right():
    interception.key_down("s")
    interception.key_down("d")
    time.sleep(1)
    interception.key_up("s")
    interception.key_up("d")
def move_backward_left():
    interception.key_down("s")
    interception.key_down("a")
    time.sleep(1)
    interception.key_up("s")
    interception.key_up("a")
def jump():
    interception.key_down("l")
    time.sleep(1)
    interception.key_up("l")
def move_slow_forward():
    interception.key_down("p")
    interception.key_down("w")
    time.sleep(1)
    interception.key_up("p")
    interception.key_up("w")

# Define abilities functions
def use_c():
    interception.key_down("c")
    interception.key_up("c")
    time.sleep(1)
    interception.left_click()
def use_q():
    interception.key_down("q")
    interception.key_up("q")
    time.sleep(1)
    randy = random.randint(0,1)
    if randy == 0:
        interception.left_click()
    else:
        interception.right_click()
def use_e():
    interception.key_down("e")
    interception.key_up("e")
    time.sleep(1)
    randy = random.randint(0,1)
    if randy == 0:
        interception.left_click()
    else:
        interception.right_click()
def use_x():
    interception.key_down("x")
    interception.key_up("x")

# Define action functions
def left_click():
    interception.left_click()
def right_click():
    interception.right_click()

# Map random numbers to corresponding actions
options = {
    0: move_forward,
    1: move_backward,
    2: move_left,
    3: move_right,
    4: move_forward_left,
    5: move_forward_right,
    6: move_backward_left,
    7: move_backward_right,
    8: jump,
    9: use_c,
    10: move_slow_forward,
    11: use_q,
    12: move_slow_forward,
    13: use_e,
    14: move_slow_forward,
    15: use_x,
}

def main():
    time.sleep(10) # Initial delay
    while True:
        rand = random.randint(0, len(options) - 1)
        options[rand]()

if __name__ == '__main__':
    main()