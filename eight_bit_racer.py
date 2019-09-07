import math
import os
import random
import time
from collections import deque
from io import BytesIO

import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops

import adb
import dqn

WIDTH = 84
HEIGHT = 84
NUM_IMAGES = 4
INPUT_SHAPE = (NUM_IMAGES, HEIGHT, WIDTH)
NUM_ACTIONS = 3

# Hyperparameters
MAX_EXPERIENCES = 5000
NUM_EPISODES = 10000
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.95
EPSILON_MIN = 0.1
EPSILON = 1

CHECKPOINT_INTERVAL = 500
SAVE_SARS_INTERVAL = 500
TRAIN = False


class Controller(object):
    def __init__(self):
        self.adb = adb.ADB()
        self.adb.open_shell()
        self.end_signal = Image.open(
            'assets/8-bit-racing/240x320/end_signal.png')
        self.end_rms = 25
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

    def right(self):
        self.adb.shell_interactive('input tap 225 300')
        return

    def left(self):
        self.adb.shell_interactive('input tap 15 290')
        return

    @staticmethod
    def stay():
        return

    def score_changed(self, score_zone):
        if not self.are_screens_similar(score_zone, self.score_zone):
            print 'SCORE CHANGED!'
            return True
        return False

    def get_reward(self, state):
        score_zone = state.crop((200, 5, 235, 55))
        if self.score_zone is None:
            self.score_zone = score_zone
        if self.is_gameover(state):
            return -1
        if self.score_changed(score_zone):
            self.score_zone = score_zone
            return 1
        return 0.1

    def get_screen(self, verbose=False):
        """Captures the screen of the device and converts to grayscale"""
        if verbose:
            start = time.time()
        screen = self.adb.shell('screencap -p')
        # ADB shell converts LF to CRLF so undo this.
        img = Image.open(BytesIO(screen.replace('\r\n', '\n')))
        if verbose:
            elapsed = time.time() - start
            print 'Took %f secs to get screen' % elapsed
        return img.convert('L')

    def get_preprocessed_screen(self, screen, verbose=False):
        if verbose:
            start = time.time()
        screen = screen.crop((35, 120, 205, 320)).resize((84, 84))
        assert screen.size == (84, 84)
        if verbose:
            elapsed = time.time() - start
            print 'Took %f secs to preprocess' % elapsed
        return screen

    def are_screens_similar(self, screen1, screen2):
        hist = ImageChops.difference(screen1, screen2).histogram()
        rms = math.sqrt(sum(map(lambda h, i: h*(i**2), hist, range(256))) / \
                (float(screen1.size[0]) * screen1.size[1]))
        if rms <= self.end_rms:
            return True
        return False

    def is_gameover(self, screen, verbose=False):
        if verbose:
            start = time.time()
        box = screen.crop((25, 200, 205, 225)).convert('L')
        gameover = self.are_screens_similar(box, self.end_signal)
        if verbose:
            elapsed = time.time() - start
            print 'Took %f secs to check gameover' % elapsed
        if gameover:
            print 'Game over!'
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


def save_sars(i, counter, best_action_name, q_sa, reward, state_processed, new_state_processed):
    state_processed.save('screens/%d-%d-0.png' % (i, counter))
    action_file = open('screens/%d-%d-action.txt' % (i, counter), 'w', 0)
    if q_sa is not None:
        action_file.write(str(q_sa) + '\n')
    action_file.write(best_action_name + ',' + str(reward) + '\n')
    action_file.close()
    new_state_processed.save('screens/%d-%d-1.png' % (i, counter))


def main():
    controller = Controller()
    # Initialize experience-replay: stores tuples (s, a, r, s')
    experiences = deque([], MAX_EXPERIENCES)
    # Initialize Q* with random weights
    epsilon = EPSILON

    if os.path.exists('model.h5'):
        print 'Loading model...'
        model = load_model('model.h5')
    else:
        model = dqn.build_model(INPUT_SHAPE, NUM_ACTIONS)

    # Stats to keep track of.
    scores = []
    scores_file = open('scores.txt', 'w', 0)
    begin_time = time.time()
    num_frames = 0

    for i in range(NUM_EPISODES):
        controller.restart_game()
        print 'Episode: %d' % i
        state = controller.get_screen()
        score = 0
        current_history = deque([], 4)
        counter = 0
        while not controller.is_gameover(state):
            # Observe state
            state = controller.get_screen()
            num_frames += 1
            state_processed = controller.get_preprocessed_screen(state)

            current_history.append(state_processed)
            combined = None
            q_sa = None
            if len(current_history) == 4:
                combined = combine_history(
                    current_history).reshape((1, 4, WIDTH, HEIGHT))

            # Select action based on epsilon-greedy (random or Q*)
            if ((random.random() < epsilon) and TRAIN) or (combined is None):
                print 'Selecting random action'
                best_action = random.randint(0, controller.num_actions-1)
            else:
                predict_start = time.time()
                q_sa = model.predict(combined)
                print 'Time to predict: %f' % (time.time() - predict_start)
                print "Q_SA", q_sa
                best_action = controller.get_actions()[np.argmax(q_sa)]

            best_action_name = controller.get_action_name(best_action)
            print best_action_name

            # Execute action and observe new state and immediate reward
            controller.execute_action(best_action)
            new_state = controller.get_screen()
            num_frames += 1
            reward = controller.get_reward(new_state)
            print 'reward: %s' % reward
            is_terminal = controller.is_gameover(new_state)
            score += reward
            new_state_processed = controller.get_preprocessed_screen(new_state)
            current_history.append(new_state_processed)
            if i % SAVE_SARS_INTERVAL == 0:
                save_sars(i, counter, best_action_name, q_sa, reward,
                          state_processed, new_state_processed)
            state = new_state
            counter += 1
            print

            if TRAIN:
                if len(current_history) == 4 and combined is not None:
                    new_combined = combine_history(
                        current_history).reshape((1, 4, WIDTH, HEIGHT))
                    # Store experience in experiences
                    experiences.append(
                        (combined, best_action, reward, new_combined, is_terminal))

                # Grab a minibatch sample from experiences
                inputs = np.zeros((BATCH_SIZE, NUM_IMAGES, WIDTH, HEIGHT))
                targets = np.zeros((BATCH_SIZE, NUM_ACTIONS))
                if len(experiences) >= BATCH_SIZE:
                    samples = random.sample(experiences, BATCH_SIZE)
                    for j, sample in enumerate(samples):
                        state_t = sample[0]
                        action = sample[1]
                        reward = sample[2]
                        state_t1 = sample[3]
                        is_terminal = sample[4]
                        inputs[j] = state_t
                        # Set target equal to final reward if terminal or immediate plus
                        # discount * max_{a'} Q*
                        targets[j] = model.predict(state_t)
                        if is_terminal:
                            targets[j, action] = reward
                        else:
                            targets[j, action] = reward + DISCOUNT_FACTOR * \
                                np.max(model.predict(state_t1))
                    train_start = time.time()
                    print 'Loss: ', model.train_on_batch(inputs, targets)
                    print 'Train Time: %f' % (time.time() - train_start)

        # Linearly decay epsilon
        if epsilon >= EPSILON_MIN:
            epsilon -= ((1-0.1)/NUM_EPISODES)
        print 'epsilon = %f' % epsilon

        scores.append(str(score))
        scores_file.write(str(score) + '\n')
        print 'Score: %d' % score

        if i % CHECKPOINT_INTERVAL == 0:
            print 'Saving model so far...'
            model.save('model.h5')
        print 'Time elapsed: %f (mins)' % ((time.time() - begin_time)/60.0)
        print 'Number of frames captured: %d' % num_frames


if __name__ == '__main__':
    main()
