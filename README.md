# BOTTI: AI Bot for First-Person Shooters (Current Version 2.0.0)

BOTTI is a fully functional AI designed to be implemented in first-person shooter games. Currently BOTTI can aim, shoot, move, look, and use abilities.

**Important:** This project builds upon [AIMi by McDaived](https://github.com/McDaived/AIMi), which provides most of the real-time object detection functionality. Credit to McDaived for this component.

## ⚠️ Disclaimer

BOTTI is not intended to simulate human gameplay in games like VALORANT or any other online game. The AI will exhibit behaviors that may be considered inhuman. **Using this in online games will likely result in a ban.** Use this program at your own risk.

Also, **THIS IS NOT A HACK!!!** This bot sucks at playing VALORANT compared to humans, so it will not be actually good at playing the game - yet. If you would like, AIMi is just the object detection and aims your mouse at enemy players it detects, so you could use that, but I'm not held responsible if you get banned and I am in no way endorsing cheating.

## Features

- **Real-Time Object Detection:** Detects enemies and aims the crosshair using YOLO-based CNN.
- **Intelligent Navigation:** Uses a Deep Q Network (DQN) for map navigation.
- **Ability Usage:** Leverages DQN for strategic ability usage in VALORANT.
- **Basic Movement:** Moves and looks around freely.
- **Automated Shooting:** Fires when an enemy is detected.

## New Feature: Deep Q Network (DQN)

I've recently implemented a Deep Q Network, a type of Convolutional Neural Network, to enhance BOTTI's capabilities:

- **Navigation:** BOTTI can now move! The DQN allows it to navigate 3D environments from a 2D screen capture. Note that this is still in early stages, so it isn't great at doing this yet.
- **Strategic Ability Usage:** BOTTI can now make intelligent decisions about when and how to use abilities. Also still in development, right now it may be randomly spamming abilities.
- **Integration with Existing Systems:** The DQN works alongside the YOLO-based object detection system, combining high-level strategy (bit of a stretch) with precise aiming and shooting (actually accurate).

NOTE: I have trained an agent using this DQN and it has been decent, however I will be improving on it further. Since GitHub doesn't allow files larger than 25MB, I can't upload the trained model (almost 500MB), so you will have to train it on your own. I recommend training it for at least 100 episodes, which for me took about 8 hours. I trained it for about 30 on VALORANT and it seemed to not get stuck, but also is just aimlessly wandering around the environment. If anyone has a solution to upload the model, please let me know. For now, you will have to train it by yourself.

## Installation

### Prerequisites

- **Interception Driver:** Many games, including VALORANT, have security measures that block Python libraries like `pyautogui` and `pynput`. To bypass these restrictions, you'll need to install the [Interception Driver](https://github.com/oblitum/Interception), which allows for direct control over input devices (keyboard and mouse).

### Steps

1. **Clone this Repository**
    ```bash
    git clone https://github.com/Cole-Godfrey/BOTTI.git
    ```
   
2. **Install the Interception Driver**  
   Follow the instructions provided in the [Interception GitHub Repository](https://github.com/oblitum/Interception) to install the driver. This is required to control the mouse and keyboard inputs directly.

3. **Install Dependencies**
    ```bash
    pip install -r assets/requirements.txt
    ```
    Please note that I am not sure if I included all libraries, so please submit an issue or pull request if the setup of BOTTI is not properly working.

4. **Start VALORANT**  
   Ensure VALORANT is running and that you're in a game or the firing range. BOTTI will begin controlling your mouse and keyboard after 10 seconds, so be ready!

5. **Run the Script**
    ```bash
    python absolutemain.py
    ```

6. **Stopping the Program**  
   This part is pretty messy. You will have to fight BOTTI for control over your mouse and keyboard if you would like to terminate the program and it has not finished training. I will implement an easier way to     stop it in the future but for now please remember this before you hit run.

## Planned Features

- **Fine-tuning of DQN:** The current method of training put simply is to reward BOTTI for making a completely different image appear on the screen. I use structural similarity to determine how similar the image                           is to the previous, and reward/punish the agent based on the result. This is currently what is causing the AI to sometimes spam abilities (since it makes the screen a lot different), so                           more than likely I will be changing the reward system in the future.
- **Graphical User Interface (GUI):** This would make it so much easier to use BOTTI. Definitely on my list of things to do, but won't do this until BOTTI actually figures out how to be competent at playing.
- **Executable Version:** To allow running the program without needing Python or an IDE. Also not gonna do this until BOTTI figures out how to play FPS games somewhat well.
- **Multi-game Support:** Adapting BOTTI for use in various FPS games. This is pretty easy since navigation and object detection can be used in any game, but not super high on my priority list.

## Contribution

Feel free to open issues or contribute to the project by submitting pull requests. Suggestions and improvements are welcome, especially in the areas of machine learning optimization and game-specific strategies! If anyone reading this knows what I should do in terms of optimizing movement.py to train BOTTI well, please tell me!

## Acknowledgements

- [McDaived](https://github.com/McDaived) for the original AIMi project
- The open-source community for various machine learning and computer vision libraries used in this project

## License

[MIT License](LICENSE)
