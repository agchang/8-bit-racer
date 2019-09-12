#!/bin/bash

EMULATOR_PATH=${EMULATOR_PATH:="emulator"}

check_for_sdk_tools() {
  if ! command -v avdmanager; then
    echo "Missing 'avdmanager' tool, please download the Android SDK!"
    exit 1
  fi

  if ! command -v emulator; then
    echo "Missing 'emulator' tool, please download the Android SDK!"
    exit 1
  fi

  if ! command -v adb; then
    echo "Missing 'adb' tool, please download the Android SDK!"
    exit 1
  fi
}

create_avd_if_necessary() {
  if ! avdmanager list avd | grep '8_bit_racer' > /dev/null; then
    sdkmanager "system-images;android-21;default;x86_64"
    avdmanager create avd -d 20 -k 'system-images;android-21;default;x86_64' --name 8_bit_racer
  else
    echo "Found existing avd."
  fi
}

# Starts the emulator and waits for boot completion.
start_avd() {
  nohup $EMULATOR_PATH -avd 8_bit_racer &
  local booted
  while [[ -z "${booted}" || "${booted}" != 1 ]]; do
    booted="$(adb shell getprop dev.bootcomplete | tr -d '\r\n')"
    echo "Waiting for emulator to boot..."
    sleep 3
  done
}

install_game() {
  adb install -r assets/8-bit-racing/8\ Bit\ Racing_v1.6_apkpure.com.apk
}

start_game() {
  adb shell am start com.londono.bitracing/com.londono.bitracing.MainMenuActivity
  echo "Ready to run!"
}

check_for_sdk_tools

if ! create_avd_if_necessary; then
  echo "Failed to create AVD!"
fi

start_avd
install_game
start_game
