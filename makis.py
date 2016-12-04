from PIL import Image
from PIL import ImageChops
from io import BytesIO
from keras.models import Sequential
from keras.models import load_model 
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, Adam
from keras.layers.convolutional import Convolution2D
from keras.initializations import normal
from collections import deque

import adb
import math
import operator
import os
import sys
import random
import time
import numpy as np
import tensorflow as tf

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

class Controller:
    def __init__(self):
        self.adb = adb.ADB()
        self.adb.OpenShell()
        self.endSignal = Image.open('assets/8-bit-racing/240x320/end_signal.png')
        self.endRms= 25
        self.numActions = 3
        self.scoreZone = None
    def executeAction(self,i):
        [self.left, self.stay, self.right][i]()
        time.sleep(0.1)
    def getActions(self):
        return range(self.numActions)
    def getActionName(self, i):
        return ["LEFT", "STAY", "RIGHT"][i]
    def right(self):
        self.adb.ShellInteractive('input tap 225 300')
        return
    def left(self):
        self.adb.ShellInteractive('input tap 15 290')
        return
    def stay(self):
        return
    def scoreChanged(self, scoreZone):
        if not self.areScreensSimilar(scoreZone, self.scoreZone):
            print 'SCORE CHANGED!'
            return True
        return False
    def getReward(self, state):
        scoreZone = state.crop((200, 5, 235, 55))
        if self.scoreZone == None:
            self.scoreZone = scoreZone
        if self.isGameover(state):
            return -1
        if self.scoreChanged(scoreZone):
            self.scoreZone = scoreZone
            return 1
        else:
            return 0.1
    def getScreen(self, verbose=False):
        """Captures the screen of the device and converts to grayscale"""
        if verbose:
            start = time.time()
        screen = self.adb.Shell('screencap -p')
        # ADB shell converts LF to CRLF so undo this.
        img = Image.open(BytesIO(screen.replace('\r\n', '\n')))
        if verbose:
            elapsed = time.time() - start
            print 'Took %f secs' % elapsed
        return img.convert('L')
    def getPreprocessedScreen(self, screen):
        screen = screen.crop((35, 120, 205, 320)).resize((84,84))
        assert screen.size == (84,84)
        return screen
    def areScreensSimilar(self, screen1, screen2):
        h = ImageChops.difference(screen1, screen2).histogram()
        rms = math.sqrt(reduce(operator.add,
                    map(lambda h, i: h*(i**2), h, range(256))
                        ) / (float(screen1.size[0]) * screen1.size[1]))
        if rms <= self.endRms:
            return True
        return False
    def isGameover(self, screen):
        box = screen.crop((25, 200, 205, 225)).convert('L')
        if self.areScreensSimilar(box, self.endSignal):
            print 'GAME OVER!'
            return True
        return False
    def restartGame(self):
        """Restarts the game from a game over state. """
        # New Game
        self.adb.ShellInteractive('input tap 120 155')
        # Play Again
        self.adb.ShellInteractive('input tap 60 215')
        time.sleep(1)
        self.adb.ShellInteractive('input tap 120 155')
        self.scoreZone = None


def combineHistory(history):
    """ Combines a history of 4 last screens into a single one. """
    assert len(history) == 4
    stacked = np.stack(history, axis=0)
    return stacked

def saveSars(i, counter, bestActionName, reward, stateProcessed, newStateProcessed):
    stateProcessed.save('screens/%d-%d-0.png' % (i,counter))
    actionFile = open('screens/%d-%d-action.txt' % (i, counter), 'w', 0)
    actionFile.write(bestActionName + ',' + str(reward))
    actionFile.close()
    newStateProcessed.save('screens/%d-%d-1.png' % (i, counter))

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
        model = dqn.buildModel(INPUT_SHAPE, NUM_ACTIONS)

    # Stats to keep track of.
    scores = []
    scoresFile = open('scores.txt', 'w', 0)
    beginTime = time.time()
    numFrames = 0

    for i in range(NUM_EPISODES):
        controller.restartGame()
        print 'Episode: %d' % i
        state = controller.getScreen()
        score = 0
        currentHistory = deque([], 4)
        counter = 0
        while not controller.isGameover(state):
            # Observe state
            state = controller.getScreen()
            numFrames+=1
            stateProcessed = controller.getPreprocessedScreen(state)

            currentHistory.append(stateProcessed)
            combined = None
            if len(currentHistory) == 4:
                combined = combineHistory(currentHistory).reshape((1, 4, WIDTH, HEIGHT))

            # Select action based on epsilon-greedy (random or Q*)
            if ((random.random() < epsilon) and TRAIN) or (combined is None):
                print 'Selecting random action'
                bestAction = random.randint(0,controller.numActions-1)
            else:
                q_sa = model.predict(combined)
                print "Q_SA", q_sa
                bestAction = controller.getActions()[np.argmax(q_sa)]

            bestActionName = controller.getActionName(bestAction) 
            print bestActionName
           
            # Execute action and observe new state and immediate reward
            controller.executeAction(bestAction)
            newState = controller.getScreen()
            numFrames+=1
            reward = controller.getReward(newState)
            print 'reward: %s' % reward
            isTerminal = controller.isGameover(newState)
            score += reward
            newStateProcessed = controller.getPreprocessedScreen(newState)
            currentHistory.append(newStateProcessed)
            if i % SAVE_SARS_INTERVAL == 0:
                saveSars(i, counter, bestActionName, reward, stateProcessed, newStateProcessed)
            state = newState
            counter+=1
            print

            if TRAIN:
                if len(currentHistory) == 4 and combined is not None:
                    newCombined = combineHistory(currentHistory).reshape((1, 4, WIDTH, HEIGHT))
                    # Store experience in experiences
                    experiences.append((combined, bestAction, reward, newCombined, isTerminal))

                # Grab a minibatch sample from experiences
                inputs = np.zeros((BATCH_SIZE, NUM_IMAGES, WIDTH, HEIGHT))
                targets = np.zeros((BATCH_SIZE, NUM_ACTIONS))
                if (len(experiences) >= BATCH_SIZE):
                    samples = random.sample(experiences, BATCH_SIZE)
                    for j,sample in enumerate(samples):
                        state_t = sample[0]
                        action = sample[1]
                        reward = sample[2]
                        state_t1 = sample[3]
                        isTerminal = sample[4]
                        inputs[j] = state_t
                        # Set target equal to final reward if terminal or immediate plus discount * max_{a'} Q*
                        targets[j] = model.predict(state_t)
                        if isTerminal:
                            targets[j, action] = reward
                        else: 
                            targets[j, action] = reward + DISCOUNT_FACTOR * np.max(model.predict(state_t1))
                    train_start = time.time()
                    print 'Loss: ', model.train_on_batch(inputs, targets)
                    print 'Train Time: %f' % (time.time() - train_start)

        # Linearly decay epsilon
        if epsilon >= EPSILON_MIN:
            epsilon -= ((1-0.1)/NUM_EPISODES)
        print 'epsilon = %f' % epsilon

        scores.append(str(score))
        scoresFile.write(str(score) + '\n')
        print 'Score: %d' % score

        if i % CHECKPOINT_INTERVAL == 0:
            print 'Saving model so far...'
            model.save('model.h5')
        print 'Time elapsed: %f (mins)' % ((time.time() - beginTime)/60.0)
        print 'Number of frames captured: %d' % numFrames

if __name__ == '__main__':
    main()
