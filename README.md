# Makis - Deep RL Agent for Android

## Dependencies
1. [Android SDK ](https://developer.android.com/studio/index.html)
  - Platform Tools and SDK Tools
  - Android API 21 Intel x86 Image

## How to run
1. ./avd.sh
2. python makis.py

## Model & Details
- The pre-trained model is already the default model (model.h5) and also present in the models/ directory.
- dqn.py defines the CNN architecture
- adb.py provides a utility class for interacting with the emulator
- makis.py contains the training algorithm
- plot.py is a utility for plotting the scores as well as producing a plot comparing performances between baseline and oracle
