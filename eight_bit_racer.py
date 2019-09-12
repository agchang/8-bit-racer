import argparse
import math
import os
import random
import time
from collections import deque
from io import BytesIO

import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageChops

import adb
import dqn

# Suppress warnings about keras using old TF apis.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

WIDTH = 84
HEIGHT = 84
NUM_IMAGES = 4
# height, width of the screenshot
IMG_DIM = (HEIGHT, WIDTH)
INPUT_SHAPE = (NUM_IMAGES, ) + IMG_DIM

# Hyperparameters
MAX_EXPERIENCES = 5000
NUM_EPISODES = 10000
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.95
EPSILON_MIN = 0.1
START_EPSILON = 1

CHECKPOINT_INTERVAL = 500
SAVE_SARS_INTERVAL = 500

DEFAULT_MODEL_FILE = 'model.h5'


class EightBitRacerController(object):
    def __init__(self):
        self.adb = adb.ADB()
        self.adb.open_shell()
        self.end_signal = Image.open(
            'assets/8-bit-racing/240x320/end_signal.png')
        self.num_actions = 3
        self.score_zone = None

    def execute_action(self, i):
        [self.left, self.stay, self.right][i]()
        time.sleep(0.1)

    def get_actions(self):
        return range(self.num_actions)

    @staticmethod
    def get_action_name(i):
        return ["LEFT", "STAY", "RIGHT"][i]

    @staticmethod
    def stay():
        return

    def right(self):
        self.adb.shell_interactive('input tap 225 300')
        return

    def left(self):
        self.adb.shell_interactive('input tap 15 290')
        return

    def _score_changed(self, score_zone):
        if not self._are_screens_similar(score_zone, self.score_zone):
            print('SCORE CHANGED!')
            return True
        return False

    def get_reward(self, state):
        score_zone = state.crop((200, 5, 235, 55))
        if self.score_zone is None:
            self.score_zone = score_zone
        if self.is_gameover(state):
            return -1
        if self._score_changed(score_zone):
            self.score_zone = score_zone
            return 1
        return 0.1

    def get_screen(self, verbose=False):
        """Captures the screen of the device and converts to grayscale"""
        if verbose:
            start = time.time()
        screen = self.adb.shell('screencap -p')
        # ADB shell converts LF to CRLF so undo this.
        img = Image.open(
            BytesIO(screen.replace('\r\n'.encode(), '\n'.encode())))
        if verbose:
            elapsed = time.time() - start
            print('Took %f secs to get screen' % elapsed)
        return img.convert('L')

    def get_preprocessed_screen(self, screen, verbose=False):
        if verbose:
            start = time.time()
        screen = screen.crop((35, 120, 205, 320)).resize(IMG_DIM)
        assert screen.size == IMG_DIM
        if verbose:
            elapsed = time.time() - start
            print('Took %f secs to preprocess' % elapsed)
        return screen

    def _are_screens_similar(self, screen1, screen2):
        hist = ImageChops.difference(screen1, screen2).histogram()
        rms = math.sqrt(sum(map(lambda h, i: h*(i**2), hist, range(256))) / \
                (float(screen1.size[0]) * screen1.size[1]))
        if rms <= 25:
            return True
        return False

    def is_gameover(self, screen, verbose=False):
        if verbose:
            start = time.time()
        box = screen.crop((25, 200, 205, 225)).convert('L')
        gameover = self._are_screens_similar(box, self.end_signal)
        if verbose:
            elapsed = time.time() - start
            print('Took %f secs to check gameover' % elapsed)
        if gameover:
            print('Game over!')
        return gameover

    def restart_game(self):
        """Restarts the game from a game over state. """
        # New Game
        self.adb.shell_interactive('input tap 120 155')
        # Play Again
        self.adb.shell_interactive('input tap 60 215')
        time.sleep(1)
        self.adb.shell_interactive('input tap 120 155')
        self.score_zone = None


def combine_history(history):
    """ Combines a history of 4 last screens into a single one. """
    assert len(history) == 4
    stacked = np.stack(history, axis=0)
    return stacked


def save_sars(episode_num, counter, best_action_name, q_sa, reward,
              state_processed, new_state_processed):
    state_processed.save('screens/%d-%d-0.png' % (episode_num, counter))
    action_file = open('screens/%d-%d-action.txt' % (episode_num, counter),
                       'w')
    if q_sa is not None:
        action_file.write(str(q_sa) + '\n')
    action_file.write(best_action_name + ',' + str(reward) + '\n')
    action_file.close()
    new_state_processed.save('screens/%d-%d-1.png' % (episode_num, counter))


def run_episode(train, controller, experiences, model, episode_num, epsilon):
    print('Episode: %d' % episode_num)

    state = controller.get_screen()
    score, counter, num_frames = 0, 0, 0
    current_history = deque([], 4)
    while not controller.is_gameover(state):
        # Observe state
        state = controller.get_screen()
        num_frames += 1
        state_processed = controller.get_preprocessed_screen(state)

        current_history.append(state_processed)
        combined = None
        q_sa = None
        if len(current_history) == 4:
            combined = combine_history(current_history).reshape(
                (1, 4, WIDTH, HEIGHT))

        # Select action based on epsilon-greedy (random or Q*)
        if ((random.random() < epsilon) and train) or (combined is None):
            print('Selecting random action')
            best_action = random.randint(0, controller.num_actions - 1)
        else:
            predict_start = time.time()
            q_sa = model.predict(combined)
            print('Time to predict: %f' % (time.time() - predict_start))
            print("Q_SA", q_sa)
            best_action = controller.get_actions()[np.argmax(q_sa)]

        best_action_name = controller.get_action_name(best_action)
        print(best_action_name)

        # Execute action and observe new state and immediate reward
        controller.execute_action(best_action)
        new_state = controller.get_screen()
        num_frames += 1
        reward = controller.get_reward(new_state)
        print('reward: %s' % reward)
        is_terminal = controller.is_gameover(new_state)
        score += reward
        new_state_processed = controller.get_preprocessed_screen(new_state)
        current_history.append(new_state_processed)
        if episode_num % SAVE_SARS_INTERVAL == 0:
            save_sars(episode_num, counter, best_action_name, q_sa, reward,
                      state_processed, new_state_processed)
        state = new_state
        counter += 1

        if args.train:
            if len(current_history) == 4 and combined is not None:
                new_combined = combine_history(current_history).reshape(
                    (1, 4, WIDTH, HEIGHT))
                # Store experience in experiences
                experiences.append(
                    (combined, best_action, reward, new_combined, is_terminal))

            # Grab a minibatch sample from experiences
            if len(experiences) >= BATCH_SIZE:
                inputs = np.zeros((BATCH_SIZE, NUM_IMAGES, WIDTH, HEIGHT))
                targets = np.zeros((BATCH_SIZE, controller.num_actions))
                samples = random.sample(experiences, BATCH_SIZE)

                for j, sample in enumerate(samples):
                    s1, action, reward, s2, is_terminal = sample
                    inputs[j] = s1
                    # Set target equal to final reward if terminal or immediate plus
                    # discount * max_{a'} Q*
                    targets[j] = model.predict(s1)
                    if is_terminal:
                        targets[j, action] = reward
                    else:
                        targets[j, action] = reward + DISCOUNT_FACTOR * \
                            np.max(model.predict(s2))
                train_start = time.time()
                print('Loss: ', model.train_on_batch(inputs, targets))
                print('Train Time: %f' % (time.time() - train_start))

    return score, num_frames


def main(args):
    controller = EightBitRacerController()

    if args.model_file is not None:
        if os.path.exists(args.model_file):
            print('Loading model...')
            model = load_model(args.model_file)
        else:
            raise Exception("Specified a non-existent model file: " +
                            args.model_file)
    else:
        model = dqn.build_model(INPUT_SHAPE, controller.num_actions)

    # Initialize experience-replay: stores tuples (s, a, r, s')
    experiences = deque([], MAX_EXPERIENCES)
    # Initialize Q* with random weights
    epsilon = args.epsilon

    # Stats to keep track of.
    scores = []
    scores_file = open('scores.txt', 'w')
    begin_time = time.time()

    for episode_num in range(NUM_EPISODES):
        controller.restart_game()
        score, num_frames = run_episode(args.train, controller, experiences,
                                        model, episode_num, epsilon)

        # Linearly decay epsilon
        if epsilon >= EPSILON_MIN:
            epsilon -= ((1 - 0.1) / NUM_EPISODES)
        print('epsilon = %f' % epsilon)

        scores.append(str(score))
        scores_file.write(str(score) + '\n')
        print('Score: %d' % score)

        if episode_num % CHECKPOINT_INTERVAL == 0:
            print('Saving model so far...')
            model.save(DEFAULT_MODEL_FILE if args.model_file is None else args.
                       model_file)

        print('Time elapsed: %f (mins)' % ((time.time() - begin_time) / 60.0))
        print('Number of frames captured: %d' % num_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs 8-bit-racer agent")
    parser.add_argument(
        '--train',
        dest='train',
        action='store_true',
        default=False,
        help=
        "Whether to train the agent or just run inference on the saved model")
    parser.add_argument(
        '--model_file',
        type=str,
        default=None,
        help=
        "If specified, resumes from a stored model file. Otherwise creates a model file at default model.h5"
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=START_EPSILON,
        help=
        "Starting epsilon value for epsilon greedy training. If train is False, this does nothing."
    )

    args = parser.parse_args()
    main(args)
