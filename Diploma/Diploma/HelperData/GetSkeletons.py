import argparse
import numpy as np
import cv2
import imutils
import os 
import time

import Estimation.Estimator as estima

#cv2.COLOR_BGR2RGB - if webcam
class VideoProcessing():
	def __init__(self, max_width=500, type_color=cv2.COLOR_BGR2GRAY, output=r'', input=r''):
		self.max_width = max_width
		self.type_color = type_color
		self.output = output
		self.input = input
		self.estimator = estima.Pose_estimation_mode()

	def video_processing(self, capture, output):
		secs = 1
		num_frames = 30
		cnt_frames = 0
		length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
		print('==========FRAMES======')
		print(length)
		print('==================')
		frame = None
		while(cnt_frames < secs * num_frames):
			if length > cnt_frames + 1:
				# Capture frame-by-frame
				ret, frame = capture.read()

				if frame is None:
					raise ConnectionError
				if ((cnt_frames + 3) % 3) == 0:
					# Our operations on the frame come here
					frame = imutils.resize(frame, width=self.max_width)
					#gray_img = cv2.cvtColor(frame, self.type_color)

					# Display the resulting frame
					#cv2.imshow('Frame', frame)

					result = self.estimator.process(frame)
					#cv2.imshow('Output after all work', result)
					name_frame = str(int(cnt_frames/3)) + '.jpg'
					path_to_save = os.path.join(output, name_frame)
					cv2.imwrite(path_to_save, result)
			cnt_frames += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break



if (__name__ == '__main__'):

	video_handler = VideoProcessing(max_width=640)

	inputs = [
				#r'E:\Project\mine\diploma\Dataset\Data\Raw\Horiz',
				#r'E:\Project\mine\diploma\Dataset\Data\Raw\Vertic', #r'E:\Project\mine\diploma\Dataset\Data\Raw\Vertic\Good'
				#r'E:\Project\mine\diploma\Dataset\Data\Raw\Other' #
				r'D:\mine\diploma\Dataset\Data\Raw\Horiz\Bad'
			 ]

	outputs = [
				#r'E:\Project\mine\diploma\Dataset\Data\Skeleton\Horiz',
				#r'E:\Project\mine\diploma\Dataset\Data\Skeleton\Vertic', #r'E:\Project\mine\diploma\Dataset\Data\Skeleton\Vertic\Good'
				#r'E:\Project\mine\diploma\Dataset\Data\Skeleton\Other' #
				r'D:\mine\diploma\test'#r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Bad'
			  ]

	cnt = 0
	time_estim = 0
	for dir_input, dir_output in zip(inputs, outputs):
		paths = os.listdir(dir_input)
		for path in paths:
			if cnt == 36:
				break
			inner_dir = path
			path = os.path.join(dir_input, path)
			if (os.path.isdir(path)):
				fileslist = os.listdir(path)
				files = [os.path.join(path, f) for f in fileslist]
				inner_path = os.path.join(dir_output, inner_dir)
				if not os.path.exists(inner_path):
					os.mkdir(inner_path)
				for file in files:
					name_dir = file.split('\\')[-1][:-4]
					more_inner_path = os.path.join(inner_path, name_dir)
					if not os.path.exists(more_inner_path):
						os.mkdir(more_inner_path)
					capture = cv2.VideoCapture(file)#default_input
					video_handler.video_processing(capture, more_inner_path)
					# When everything done, release the capture
					capture.release()
					cv2.destroyAllWindows()
			else:
				cnt += 1
				inner_dir = path.split('\\')[-1][:-4]
				path = os.path.join(dir_input, path)
				inner_path = os.path.join(dir_output, inner_dir)
				if not os.path.exists(inner_path):
					os.mkdir(inner_path)
				capture = cv2.VideoCapture(path)#default_input
				start = time.time()
				video_handler.video_processing(capture, inner_path)
				end = time.time()
				time_estim += end - start
				# When everything done, release the capture
				capture.release()
				cv2.destroyAllWindows()
	print('time estimator: {} sec'.format(time_estim / cnt))


