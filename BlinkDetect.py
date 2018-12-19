# import the packages
from scipy.spatial import distance as dist
from imutils import face_utils
import datetime
import numpy as np
import imutils
import os
import multiprocessing as mp
#import queue
import time
import dlib
import cv2
import VideoCaptureAsync
import struct

# from imutils.video import VideoStream
# import argparse

class ConstsClass:
	# eye aspect ratio for a blink
	# number of consecutive frames the eye must be below the threshold
	EYE_AR_THRESH = 0.3
	EYE_AR_CONSEC_FRAMES = 3
	FrameWidth = 720
	NumOfFramesToAverage = 10
	
	@staticmethod
	def CalcEAR(eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])
	
		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])
	
		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)
	
		# return the eye aspect ratio
		return ear


class ImageProcUtil:

	def ImproveLowLight_Average(self, frameIn, vs, NumFramesToAverage):
	
		average_frame = np.zeros(frameIn.shape, dtype=float)
		average_frame += frameIn
		for i in range(1, NumFramesToAverage - 1):
			(grabbed, frame) = vs.read()
			if grabbed:
				frame = imutils.resize(frame, width=frameIn.shape[1])
			else:
				continue
			average_frame += frame

		average_frame /= NumFramesToAverage
		average_frame.astype(np.uint8)
		average_normalized_frame = cv2.normalize(average_frame, None, 0, 255, cv2.NORM_MINMAX)
		frame = average_normalized_frame.astype(np.uint8)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return gray
	
	def ImproveLowLight_CLAHE(self, frame):
	
		if len(frame.shape) == 3:
			gray_before_CLAHE = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif len(frame.shape) == 2:
			gray_before_CLAHE = frame

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		gray_after_CLAHE = clahe.apply(gray_before_CLAHE)

		if gray_after_CLAHE.dtype==np.uint16:
			gray_after_CLAHE=np.uint8(cv2.normalize(gray_after_CLAHE, None, 0, 255, cv2.NORM_MINMAX))

		return gray_after_CLAHE

	def ImproveLowLight_CLAHE_Average(self, frameIn,vs, NumFramesToAverage):

		frame_after_clahe = self.ImproveLowLight_CLAHE(frameIn)
		average_frame = np.zeros(frame_after_clahe.shape, dtype=float)
		average_frame += frame_after_clahe
		for i in range(1, NumFramesToAverage - 1):
			(grabbed, frame) = vs.read()
			if grabbed:
				frame = imutils.resize(frame, width=frameIn.shape[1])
				frame_after_clahe=self.ImproveLowLight_CLAHE(frame)
			else:
				continue
			average_frame += frame_after_clahe

		average_frame /= NumFramesToAverage
		average_frame.astype(np.uint8)
		average_normalized_frame = cv2.normalize(average_frame, None, 0, 255, cv2.NORM_MINMAX)
		return average_normalized_frame.astype(np.uint8)



	def HandleEyeDetection(self, leftEye, rightEye):
	
		leftEAR = ConstsClass.CalcEAR(leftEye)
		rightEAR = ConstsClass.CalcEAR(rightEye)
	
		# average the eye aspect ratio together for both eyes
		avgEAR = (leftEAR + rightEAR) / 2.0
	
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
	
		return leftEyeHull, rightEyeHull, avgEAR


def main():

	if __name__ == '__main__':
		print(__name__)
		# initialize the frame counters and the total number of blinks
		Local_Closed_Counter = 0
		TOTAL = 0
		frameNum = 0
		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		print("[INFO] loading facial landmark predictor...")
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # args["shape_predictor"]

		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# start the video stream thread
		print("[INFO] starting video stream thread...")
		# vs = FileVideoStream(args["video"]).start()

		# vs = cv2.VideoCapture(0)

		vcapAsync = VideoCaptureAsync.VideoCaptureAsync()
		vcapAsync.start()

		time.sleep(3) #give time for the async grabber to wake up and grab

		#(grabbed, frame) = vcapAsync.read()
		# read from files
		#(frame, grabbed) = ReadRawFile(frameNum)
		(frame, grabbed) = ReadPngFile(frameNum)

		if not grabbed:
			print("[INFO] bad format frame, exiting")
			quit()

		darkField = cv2.imread('dark.png')
		darkField = imutils.resize(darkField, width=ConstsClass.FrameWidth)

		frame = imutils.resize(frame, width=ConstsClass.FrameWidth)

		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		save = True
		if save:
			out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1] * 2, frame.shape[0]))

		UtilsClass = ImageProcUtil()

		method = ['Averaging', 'Clahe', 'Averaging then Clahe', 'Clahe then Averaging', 'HSV v component and Clahe']


		# loop over frames from the video stream
		while True:

			fpsStart =datetime.datetime.now()


			#(grabbed, frame) = vcapAsync.read()
			#(frame, grabbed) = ReadRawFile(frameNum)
			(frame, grabbed) = ReadPngFile(frameNum)
			if not grabbed:
				print("[INFO] bad format frame num: "+str(frameNum)+"received, continuing")
				continue

			# SaveDarkField(Q, frame)
			frameNum += 1
			frame = imutils.resize(frame, width=ConstsClass.FrameWidth)

			#frame -= darkField  # remove dark field noise

			# loop through methods
			#improve_low_light = int((frameNum / 40) % (len(method))) + 1  # change the method every 80 frames
			improve_low_light =2
			if improve_low_light <0:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			elif improve_low_light == 1:
				gray = UtilsClass.ImproveLowLight_Average(frame, vcapAsync, ConstsClass.NumOfFramesToAverage)

			elif improve_low_light == 2:
				gray = UtilsClass.ImproveLowLight_CLAHE(frame)

			elif improve_low_light == 3:
				gray = UtilsClass.ImproveLowLight_Average(frame, vcapAsync, ConstsClass.NumOfFramesToAverage)
				gray = UtilsClass.ImproveLowLight_CLAHE(gray)

			# todo: refactor averaging so it doesnt depend on vcap/vs and can ve fed by CLAHE
			elif improve_low_light == 4:
				gray = UtilsClass.ImproveLowLight_CLAHE_Average(frame, vcapAsync, ConstsClass.NumOfFramesToAverage)

			elif improve_low_light == 5:
				HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				v_only = HSV[:, :, 2]
				gray = v_only
				gray = UtilsClass.ImproveLowLight_CLAHE(gray)

			else:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			frameBGR = np.uint8(cv2.normalize(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), None, 0, 255, cv2.NORM_MINMAX))

			# detect faces in the grayscale frame
			eyeRects = detector(gray, 0)
			for rect in eyeRects:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]

				(leftEyeHull, rightEyeHull, ear) = UtilsClass.HandleEyeDetection(leftEye, rightEye)
				cv2.drawContours(frameBGR, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frameBGR, [rightEyeHull], -1, (0, 255, 0), 1)

				# check to see if the eye aspect ratio is below the blink
				# threshold, and if so, increment the blink frame counter
				if ear < ConstsClass.EYE_AR_THRESH:
					Local_Closed_Counter += 1

				# otherwise, the eye aspect ratio is not below the blink
				# threshold
				else:
					# reset the eye frame counter
					Local_Closed_Counter = 0

				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if Local_Closed_Counter >= ConstsClass.EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
					# reset the eye frame counter
					Local_Closed_Counter = 0

				# draw the total number of blinks on the frame along with
				# the computed eye aspect ratio for the frame

				cv2.putText(frameBGR, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.putText(frameBGR, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

			fpsStop=datetime.datetime.now()
			fpsElapsed=(fpsStop-fpsStart).total_seconds()*1000

			cv2.putText(frameBGR, "Elap: {:.1f} ms".format(fpsElapsed), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
			            (255, 255, 255), 1)
			cv2.putText(frameBGR, "Q size: {:d}".format(vcapAsync.quelen()), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
			            (255, 255, 255), 1)
			# show the frame
			# grayBGR = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
			grayBGR = np.stack([gray, gray, gray], 2)

			cv2.putText(grayBGR, "improved frame", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			cv2.putText(grayBGR, method[improve_low_light - 1], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
			            (255, 255, 255), 1)


			bothImages = cv2.hconcat([frameBGR, grayBGR])

			# cv2.imshow("Frame", frame)
			cv2.imshow("Frame", bothImages)

			if save:
				out.write(bothImages)

			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		# do a bit of cleanup

		vcapAsync.stop()
		if save:
			out.release()
		cv2.destroyAllWindows()

def ReadPngFile(frameNum):
	filePath = r"E:\BlinkDetect\ZWO_camera_input\14_54_22_Png\\"
	fileList = os.listdir(filePath)
	if frameNum<len(fileList):
		frame = cv2.imread(filePath + fileList[frameNum],-cv2.IMREAD_ANYDEPTH)
		grabbed = True
	else:
		frame=None
		grabbed=False

	return frame, grabbed

def ReadRawFile(frameNum):
	filePath = r"E:\BlinkDetect\ZWO_camera_input\14_54_22_Raw\\"
	fileList = os.listdir(filePath)

	frame=np.reshape(np.fromfile(filePath + fileList[frameNum], dtype=np.dtype('<H')),[960,1280]);
	grabbed=True

	fr8bit = np.uint8(cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX))

	cv2.imshow("uint -8", fr8bit)
	cv2.waitKey(0)
	# if the `q` key was pressed, break from the loop

	return frame, grabbed


#for occasional use
def SaveDarkField(Q, frame):
	Q.put(frame)
	averageDarkFrame = np.zeros(frame.shape, dtype=float)
	if Q.full():
		for i in range(1, 100):
			frameInQ = Q.get()
			averageDarkFrame += frameInQ
		averageDarkFrame /= 100
		averageDarkFrame.astype(np.uint8)
		cv2.imwrite('dark.png', averageDarkFrame)


main()
