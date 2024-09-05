import time, interception, random
interception.auto_capture_devices()

def look():
    start_time = time.time()
    randx = random.randint(-2000, 2000)
    stepx = randx / 60
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            break
        interception.move_relative(int(stepx), 0)
        time.sleep(1/60)

def shake():
    start_time = time.time()
    randx = random.randint(-50, 50)
    stepx = randx / 5
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 5/60:
            break
        interception.move_relative(int(stepx), 0)
        time.sleep(1/60)

time.sleep(10)
while True:
    time.sleep(3)
    look()


