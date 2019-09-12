# 8-bit-racer - Deep RL Agent for Android

## Dependencies
1. [Android SDK ](https://developer.android.com/studio/index.html)
  - Platform Tools and SDK Tools
  - Android API 21 Intel x86\_64 Image, i.e: "system-images;android-21;default;x86\_64"\

    Note avd.sh will try to download the system image and create the avd for you, but if this
    doesn't work, you can try using Android Studio to manually download and create the image and avd.
2. Tested on python 3.6.8

## How to run
1. Run the Android emulator 
  - `./avd.sh`
  - If you have issues getting the emulator to run like I do, due to issues with having both the Android Studio managed SDK
    and the command line tools downloaded separately, you can specify an environment variable to point at an emulator path, i.e.:

    `EMULATOR_PATH=~/Android/Sdk/emulator/emulator ./avd.sh`
2. `python eight_bit_racer.py`

## Model & Details
- The pre-trained model is already the default model (model.h5) and also present in the models/ directory.
- dqn.py defines the CNN architecture
- adb.py provides a utility class for interacting with the emulator
- eight\_bit\_racer.py contains the training algorithm
- plot.py is a utility for plotting the scores as well as producing a plot comparing performances between baseline and oracle
