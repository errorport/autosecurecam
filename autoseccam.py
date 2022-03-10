import cv2
import sys
from datetime import datetime
from synthesizer import Player, Synthesizer, Waveform

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

player = Player()
player.open_stream()
synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)

def beeper():
	player.play_wave(synthesizer.generate_constant_wave(4400.0, 0.01))
	player.play_wave(synthesizer.generate_constant_wave(2400.0, 0.05))

if __name__ == "__main__":

	video_capture = cv2.VideoCapture(0)
	while True:
		ret, frame = video_capture.read()
		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			faces = face_cascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(10, 10),
				flags = cv2.CASCADE_SCALE_IMAGE
			)

			for (x, y, w, h) in faces:
				now = datetime.now()
				current_time = now.strftime("%H:%M:%S")
				print("Detection: ", current_time)
				cv2.imwrite("detections/"+current_time+".jpg", frame)
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
				beeper()

			# Display the resulting frame
			cv2.imshow('Video', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	video_capture.release()
	cv2.destroyAllWindows()
