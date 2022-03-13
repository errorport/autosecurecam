import cv2
import sys
import threading
import time
import logging
from enum import Enum
from datetime import datetime
from synthesizer import Player, Synthesizer, Waveform

from config import *

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

player = Player()
player.open_stream()
player_beeper = Player()
player_beeper.open_stream()

synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)

class State(Enum):
	STANDBY = 1
	ALARM = 2

ALARM_STATE = State.STANDBY

def arming_sequence():
	logging.info("Arming...")
	for counter in range(0, ARMING_COUNTER_A):
		player.play_wave(synthesizer.generate_chord([880.0, 920.0], 0.5))
		player.play_wave(synthesizer.generate_chord([1.0], 2.0))
		time.sleep(1)
	for counter in range(0, ARMING_COUNTER_B):
		player.play_wave(synthesizer.generate_chord([940.0, 1100.0], 0.05))
		player.play_wave(synthesizer.generate_chord([1.0], 0.5))
	logging.info("Armed.")

def standby_sequence():
	while True:
		if ALARM_STATE == State.STANDBY:
			logging.info("Standing by...")
			player.play_wave(synthesizer.generate_chord([440.0], 0.5))
			player.play_wave(synthesizer.generate_chord([240.0], 0.5))
			player.play_wave(synthesizer.generate_chord([140.0], 0.5))
			player.play_wave(synthesizer.generate_chord([80.0], 0.5))
			while ALARM_STATE == State.STANDBY:
				player.play_wave(synthesizer.generate_chord([240.0], 1))
				time.sleep(STANDBY_SIGNAL_DELAY)

def alarm_sequence():
	global ALARM_STATE
	while True:
		if ALARM_STATE == State.ALARM:
			logging.info("Detection alarm!")
			for counter in range(0, ALARM_COUNTER):
				player.play_wave(synthesizer.generate_chord([440.0, 420.0], 1.0))
				player.play_wave(synthesizer.generate_chord([440.0, 420.0, 8000.0, 12000.0], 1.0))
				player.play_wave(synthesizer.generate_chord([440.0, 420.0], 1.0))
				time.sleep(ALARM_SIGNAL_DELAY)
			ALARM_STATE = State.STANDBY

def detection_beeper():
	player_beeper.play_wave(synthesizer.generate_constant_wave(4400.0, 0.01))
	player_beeper.play_wave(synthesizer.generate_constant_wave(2400.0, 0.01))
	player_beeper.play_wave(synthesizer.generate_constant_wave(5400.0, 0.01))

if __name__ == "__main__":
	logging.basicConfig(filename=LOGFILE, encoding='utf-8', format=LOG_FMT, level=logging.INFO, datefmt="%Y-%m-%d_%H:%M:%S")
	standby_thread = threading.Thread(target=standby_sequence)
	alarm_thread = threading.Thread(target=alarm_sequence)
	arming_sequence()
	standby_thread.start()
	alarm_thread.start()
	video_capture = cv2.VideoCapture(0)
	while True:
		ret, frame = video_capture.read()
		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

			faces = face_cascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(10, 10),
				flags = cv2.CASCADE_SCALE_IMAGE
			)

			for (x, y, w, h) in faces:
				ALARM_STATE = State.ALARM
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
				now = datetime.now()
				current_time = now.strftime("detection_%Y-%m-%d_%H:%M:%S")
				logging.info("Detection: ", current_time)
				cv2.imwrite("detections/"+current_time+".jpg", frame)
				detection_beeper()

			if DISPLAY:
				cv2.imshow('Video', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	video_capture.release()
	cv2.destroyAllWindows()
