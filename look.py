import time
import interception
import random

def move_mouse_relative(dx, dy):
    """
    Moves the mouse cursor relative to its current position.
    Args:
        dx (int): Change in X-coordinate.
        dy (int): Change in Y-coordinate.
    """
    interception.move_relative(dx, dy)
    time.sleep(1/60)

def look():
    """
    Simulates subtle random mouse movement.
    """
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            break
        rand_x = random.randint(-10, 10)  # Adjust the range as needed
        rand_y = random.randint(-10, 10)
        move_mouse_relative(rand_x, rand_y)

def shake():
    """
    Simulates a jittery mouse movement.
    """
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 5/60:
            break
        rand_x = random.randint(-5, 5)  # Adjust the range as needed
        rand_y = random.randint(-5, 5)
        move_mouse_relative(rand_x, rand_y)

def main():
    try:
        interception.auto_capture_devices()
        time.sleep(10)  # Initial delay before starting movement
        while True:
            time.sleep(3)
            look()
            shake()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


