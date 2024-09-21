import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import mss
import time
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import interception
import logging

# ==========================
# Setup Logging
# ==========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ==========================
# Initialize Interception
# ==========================
interception.auto_capture_devices()

# ==========================
# Action History
# ==========================
action_history = deque(maxlen=100)


# ==========================
# Frame Stacking Class
# ==========================
class FrameStacker:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84))
        for _ in range(self.stack_size):
            self.frames.append(gray)
        return self._get_stacked_frames()

    def append(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84))
        self.frames.append(gray)
        return self._get_stacked_frames()

    def _get_stacked_frames(self):
        # Normalize the frames
        normalized_frames = [frame / 255.0 for frame in self.frames]
        return np.stack(normalized_frames, axis=0)


frame_stacker = FrameStacker(stack_size=4)


# ==========================
# Screen Capture Function
# ==========================
def get_frame():
    try:
        with mss.mss() as sct:
            # Adjust monitor settings if necessary
            monitor = {"top": 40, "left": 0, "width": 1920, "height": 1080}
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)  # Translate from image to numpy array
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # RGBA to RGB for NN
            return frame
    except Exception as e:
        logging.error(f"Error capturing frame: {e}")
        return None


# ==========================
# Action Functions
# ==========================
def move_forward():
    interception.key_down('w')
    time.sleep(1)
    interception.key_up('w')
    logging.debug("Action: Move Forward")


def move_backward():
    interception.key_down('s')
    time.sleep(1)
    interception.key_up('s')
    logging.debug("Action: Move Backward")


def move_left():
    interception.key_down('a')
    time.sleep(1)
    interception.key_up('a')
    logging.debug("Action: Move Left")


def move_right():
    interception.key_down('d')
    time.sleep(1)
    interception.key_up('d')
    logging.debug("Action: Move Right")


def look_left():
    total_distance = -690
    steps = 50  # Reduced steps = smoother
    delay = 0.005

    dx = total_distance / steps

    for _ in range(steps):
        interception.move_relative(int(dx), 0)
        time.sleep(delay)
    logging.debug("Action: Look Left")


def look_right():
    total_distance = 690
    steps = 50
    delay = 0.005

    dx = total_distance / steps

    for _ in range(steps):
        interception.move_relative(int(dx), 0)
        time.sleep(delay)
    logging.debug("Action: Look Right")


def look_up():
    total_distance = -30
    steps = 50
    delay = 0.005

    dy = total_distance / steps

    for _ in range(steps):
        interception.move_relative(0, int(dy))
        time.sleep(delay)
    logging.debug("Action: Look Up")


def look_down():
    total_distance = 30
    steps = 50
    delay = 0.005

    dy = total_distance / steps

    for _ in range(steps):
        interception.move_relative(0, int(dy))
        time.sleep(delay)
    logging.debug("Action: Look Down")


def use_c():
    interception.key_down('c')
    time.sleep(0.1)  # Short press
    interception.key_up('c')
    time.sleep(1)
    interception.left_click()
    logging.debug("Action: Use C Ability")


def use_e():
    interception.key_down('e')
    time.sleep(0.1)
    interception.key_up('e')
    time.sleep(1)
    interception.left_click()
    logging.debug("Action: Use E Ability")


def use_q():
    interception.key_down('q')
    time.sleep(0.1)
    interception.key_up('q')
    time.sleep(1)
    interception.left_click()
    logging.debug("Action: Use Q Ability")


def use_x():
    interception.key_down('x')
    time.sleep(0.1)
    interception.key_up('x')
    logging.debug("Action: Use X Ability")


# ==========================
# Action Mapping
# ==========================
ACTION_MAP = {
    0: move_forward,
    1: move_backward,
    2: move_left,
    3: move_right,
    4: look_left,
    5: look_right,
    6: look_up,
    7: look_down,
    8: use_c,
    9: use_e,
    10: use_q,
    11: use_x
}


# ==========================
# CNN Model Definition
# ==========================
class CNNModel(nn.Module):
    def __init__(self, input_channels=4, num_actions=12):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), 
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==========================
# DQN Agent Definition
# ==========================
class DQNAgent:
    def __init__(self, input_channels=4, num_actions=12, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=50000):
        self.num_actions = num_actions
        self.model = CNNModel(input_channels, num_actions).to(device)
        self.target_model = CNNModel(input_channels, num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set target model to evaluation mode
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.loss_fn = nn.MSELoss()
        self.batch_size = 64
        self.update_target_every = 1000  #steps
        self.step_count = 0
        self.losses = []

    def act(self, state):
        if random.random() <= self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            logging.debug(f"Exploring: Action {action}")
            return action
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # add batch dimension
        with torch.no_grad():
            q_values = self.model(state)
        action = q_values.argmax().item()
        logging.debug(f"Exploiting: Action {action}")
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        logging.debug(f"Remembered: Action {action}, Reward {reward}, Done {done}")

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Current Q values
        current_q = self.model(states).gather(1, actions)

        # Next Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)

        # Compute target Q values
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        #compute loss
        loss = self.loss_fn(current_q, target_q)
        self.losses.append(loss.item())

        # Optimize Model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient Clipping
        self.optimizer.step()

        #epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_model()
            logging.info("Target model updated.")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': list(self.memory),
            'epsilon': self.epsilon
        }, path)
        logging.info(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.memory = deque(checkpoint['memory'], maxlen=self.memory.maxlen)
            self.epsilon = checkpoint['epsilon']
            logging.info(f"Model loaded from {path}")
        else:
            logging.warning(f"No saved model found at {path}")


# ==========================
# Reward Function
# ==========================
def calculate_reward(state, next_state, action, step):
    # Calculate structural similarity index with data_range=1.0
    ssim_index, _ = ssim(next_state, state, full=True, data_range=1.0)

    # Calculate the difference between the two states
    diff = cv2.absdiff(next_state, state)
    change = np.sum(diff)

    # Initialize reward
    reward = 0

    #Reward for movement (change in the environment)
    if change > 5000:
        reward += 1
    else:
        reward -= 0.1  # Penalty for not moving

    #penalty for very similar states (to avoid getting stuck)
    if ssim_index > 0.98:
        reward -= 0.5

    # Reward for exploring new areas (lower similarity is better)
    reward += (1 - ssim_index) * 2

    # stronger penalty for repetitive actions
    if len(action_history) >= 10 and action_history.count(action) > 7:
        reward -= 1.5  # Increased penalty

    # Add action to history
    action_history.append(action)

    # clip reward to avoid extreme values
    return np.clip(reward, -1, 1)


# ==========================
# Episode Termination Check
# ==========================
def is_episode_done(state, next_state, step):
    return step >= 1000


# ==========================
# Logger Class
# ==========================
class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.rewards = []
        self.epsilons = []
        self.losses = []
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'screenshots'), exist_ok=True)

    def log_episode(self, episode, reward, epsilon, loss):
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.losses.append(loss)

        logging.info(f"Episode: {episode}, Reward: {reward:.2f}, Epsilon: {epsilon:.4f}, Loss: {loss:.4f}")

        if episode % 10 == 0 and episode != 0:
            self.plot_metrics()

    def log_screenshot(self, episode, frame):
        filename = os.path.join(self.log_dir, 'screenshots', f'episode_{episode}.png')
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        logging.info(f"Screenshot saved for Episode {episode}")

    def plot_metrics(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.plot(self.rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.subplot(132)
        plt.plot(self.epsilons)
        plt.title('Epsilon per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')

        plt.subplot(133)
        plt.plot(self.losses)
        plt.title('Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'metrics.png'))
        plt.close()
        logging.info("Metrics plotted and saved.")


# ==========================
# Main Training Loop
# ==========================
def train(episodes, steps_per_episode, save_path, load_path=None):
    agent = DQNAgent()

    if load_path:
        agent.load(load_path)
        logging.info("Saved agent loaded.")

    logger = Logger(f'val_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    for episode in range(1, episodes + 1):
        frame = get_frame()
        if frame is None:
            logging.warning("No frame captured, skipping episode.")
            continue
        state = frame_stacker.reset(frame)

        total_reward = 0
        episode_loss = []

        logging.info(f"Starting Episode {episode}")

        for step in range(steps_per_episode):
            action = agent.act(state)

            #Perform selected action
            if action in ACTION_MAP:
                ACTION_MAP[action]()
            else:
                logging.warning(f"Invalid action {action}")

            #Capture next frame
            next_frame = get_frame()
            if next_frame is None:
                logging.warning("No next frame captured, ending episode.")
                done = True
                break
            next_state = frame_stacker.append(next_frame)

            # Calculate reward
            reward = calculate_reward(state[-1], next_state[-1], action, step)
            total_reward += reward

            # check if episode is done
            done = is_episode_done(state, next_state, step)

            #store experience
            agent.remember(state, action, reward, next_state, done)

            #replay experiences
            agent.replay()

            state = next_state

            if done:
                logging.info(f"Episode {episode} done at step {step}")
                break

        #every 10 episodes, update target model and save
        if episode % 10 == 0:
            logger.log_screenshot(episode, frame)
            agent.save(f"{save_path}_episode_{episode}.pth")

        #log metrics
        avg_loss = np.mean(agent.losses) if agent.losses else 0
        logger.log_episode(episode, total_reward, agent.epsilon, avg_loss)
        agent.losses = []  #reset losses for next episode

    #save final model
    agent.save(f"{save_path}_final.pth")
    logger.plot_metrics()
    logging.info("Training completed.")


# ==========================
# Device Configuration
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ==========================
# Run Training
# ==========================
if __name__ == "__main__":
    save_dir = "val_saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "val_dqn_model")
    load_path = None  #set specific path if you want to load saved model

    #adjust episodes and steps_per_episode as needed
    train(episodes=10, steps_per_episode=50, save_path=save_path, load_path=load_path)
