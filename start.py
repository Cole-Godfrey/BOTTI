import subprocess
import os
import interception
interception.move_to(100,100)

subprocess.call(["pip", "install", "-r", "assets/requirements.txt"])
from utils import inputs
os.system('cls' if os.name == 'nt' else 'clear')
print(r'''
Welcome to VALBOT!
Made by Cole Godfrey - built off of AIMi
NOTE: A lot of this code was from https://github.com/McDaived/AIMi.
I have made minor changes to the detect.py and added scripts for looking and using abilities,
but almost all of the real-time object detection was by McDaived.
''')
print('\033[1;35m[Control] ->','\033[1;34m[F4] ENABLED: Always On/Hold Mode')
print('\033[1;35m          ->','\033[1;34m[Mouse4] Hold Mode: Press/Release')
print('\033[1;35m          ->','\033[1;34m[0] Exit')
print('\033[1;33m\n[Tips] Press F4 to set mode\n')
interception.auto_capture_devices()
inputs.aimbot(ENABLE_AIMBOT=True)