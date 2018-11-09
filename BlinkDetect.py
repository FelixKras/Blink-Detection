
# import the packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

class ConstsClass:
	# eye aspect ratio for a blink
	# number of consecutive frames the eye must be below the threshold
	EYE_AR_THRESH = 0.3
	EYE_AR_CONSEC_FRAMES = 3
	FrameWidth=720

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

	def ImproveLowLight_Average(self, frameIn):

		NumFramesToAverage = 3
		average_frame = np.zeros(frameIn.shape,dtype=float)
		average_frame += frameIn
		for i in range(1, NumFramesToAverage - 1):
			(grabbed, frame) = vs.read()
			frame = imutils.resize(frame, width=frameIn.shape[1])
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

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
		gray_after_CLAHE = clahe.apply(gray_before_CLAHE)
		return gray_after_CLAHE

	def HandleEyeDetection(self,leftEye,rightEye):

		leftEAR = ConstsClass.CalcEAR(leftEye)
		rightEAR = ConstsClass.CalcEAR(rightEye)

		# average the eye aspect ratio together for both eyes
		avgEAR = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		return leftEyeHull,rightEyeHull,avgEAR

# initialize the frame counters and the total number of blinks
Local_Closed_Counter = 0
TOTAL = 0

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

fileStream = False
#vs = VideoStream().start()
vs = cv2.VideoCapture(0)

#vs = cv2.VideoCapture(args["video"])
#vs = VideoStream(usePiCamera=True).start()

time.sleep(0.5)

(grabed,frame)=vs.read()
if frame is None:
	print("[INFO] bad format frame, exiting")
	quit()

frame = imutils.resize(frame, width=ConstsClass.FrameWidth)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
save=True
if save:
	out = cv2.VideoWriter('output.avi',fourcc, 20.0,(frame.shape[1],frame.shape[0]))

UtilsClass= ImageProcUtil()

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	fps = imutils.video.FPS().start()

	(grabed,frame) = vs.read()

	if frame is None:
		print("[INFO] bad format frame, exiting")
		break

	frame = imutils.resize(frame, width=ConstsClass.FrameWidth)

	improve_low_light=2
	if improve_low_light==1:
		gray=UtilsClass.ImproveLowLight_Average(frame)
	elif improve_low_light==2:
		gray=UtilsClass.ImproveLowLight_CLAHE(frame)
	elif improve_low_light==3:
		gray = UtilsClass.ImproveLowLight_Average(frame)
		gray = UtilsClass.ImproveLowLight_CLAHE(gray)
	else:
		gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		(leftEyeHull,rightEyeHull,ear)=UtilsClass.HandleEyeDetection(leftEye,rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

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
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	fps.stop()
	cv2.putText(frame, "FPS: {:.2f}".format(1/fps.elapsed()), (10, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

	# show the frame

	bothImages=cv2.hconcat(frame,gray)

	#cv2.imshow("Frame", frame)
	cv2.imshow("Frame", bothImages)

	if save:
		out.write(frame)

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
vs.release()
out.release()
cv2.destroyAllWindows()
#vs.stop()
