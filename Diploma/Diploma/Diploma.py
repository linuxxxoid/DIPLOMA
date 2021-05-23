
import argparse
import numpy as np
import cv2
import imutils
import os 

import Estimation.Estimator as estima

#cv2.COLOR_BGR2RGB - if webcam
class Video_processing_mode():
	def __init__(self, max_width=500, type_color=cv2.COLOR_BGR2GRAY, output=r'', input=r''):
		self.max_width = max_width
		self.type_color = type_color
		self.output = output
		self.input = input
		self.estimator = estima.Pose_estimation_mode()

	def video_processing(self, capture):
		secs = 1
		num_frames = 30
		cnt_frames = 0
		while(cnt_frames < secs * num_frames):
			# Capture frame-by-frame
			ret, frame = capture.read()

			if frame is None:
				raise ConnectionError

			# Our operations on the frame come here
			frame = imutils.resize(frame, width=self.max_width)
			#gray_img = cv2.cvtColor(frame, self.type_color)

			# Display the resulting frame
			cv2.imshow('Frame', frame)

			result = self.estimator.process(frame)
			cv2.imshow('Output after all work', result)
			name_frame = self.input + str(cnt_frames) + '.jpg'
			path_to_save = os.path.join(self.output, name_frame)
			cv2.imwrite(path_to_save, result)
			cnt_frames += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break



if (__name__ == '__main__'):
	parser = argparse.ArgumentParser()
	parser.add_argument('--webcam', type=str, default='False')
	parser.add_argument('--output', type=str, default=r'E:\Project\mine\diploma\Dataset\Softmax\Video\Output')
	args = vars(parser.parse_args())

	video_handler = Video_processing_mode(max_width=360,
									output=args['output'],
									input = 'f')


	if args['webcam'] == 'True':      
		capture = cv2.VideoCapture(0)
	else:
		default_input = r'E:\Project\mine\diploma\Dataset\Softmax\Video\Input\f.mp4'
		capture = cv2.VideoCapture(default_input)

	video_handler.video_processing(capture)
	# When everything done, release the capture
	capture.release()
	cv2.destroyAllWindows()
