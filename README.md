BOTTI is a fully functional AI that can be implemented for any first-person shooter (although it does so astoundingly horribly). It has real-time object detection that allows it to point the crosshair and shoot at enemies. Also has basic movement and free looking. NOTE: This was built off of https://github.com/McDaived/AIMi, so credits to him for most of the real-time object detection.

WARNING
This program is not meant to simulate a human playing VALORANT or any other game. This program will lock on and shake around enemies it sees. Because of this, if you go into a game, there is a VERY HIGH CHANCE THAT YOU WILL GET BANNED if you attempt to use this in an online game, so use this at your own risk.

To run this project, you will need to do the following:

1. Clone this repository
2. Install interception driver - since VALORANT (and probably other games) has security measures to prevent pyautogui and pynput from working, you need to install the interception driver, which essentially allows us to directly control input devices so that we can execute key presses and mouse movement as if we were typing/moving the devices literally. It can be found at https://github.com/oblitum/Interception
3. MAKE SURE VALORANT IS ALREADY LOADED AND IN A GAME/FIRING RANGE! Once you run the main script, after 10 seconds your mouse and keyboard will be moving on its own (inputs sent by move.py and look.py), so if you are not in the game you will be typing random shit and your mouse will be moving across the screen.
4. run absolutemain.py

To stop the program it is kinda messy. Since it is moving your mouse and keyboard, you will have to fight against it to stop absolutemain.py in your IDE. I plan on implementing a quit command eventually.

I may add a GUI or make this an executable in the future for easier usage, but right now I'm focused on making BOTTI become a somewhat capable player.
