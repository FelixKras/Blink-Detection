#import threading
import statistics
import multiprocessing as mp
import cv2
import os
import datetime
import time

class VideoCaptureAsync:
	def __init__(self,src=0):
		self.src=src
		self.started = False
		self.FramesQue=mp.Queue(20)
		#self.read_lock = threading.Lock()

	def set(self, var1, var2):
		self.cap.set(var1, var2)
	
	def start(self):
		if self.started:
			print('[!] Asynchroneous video capturing has already been started.')
			return None

		self.started = True

		#self.thread = threading.Thread(target=self.update, args=())
		#self.thread.daemon = True
		#self.thread.start()

		p = mp.Process(target=self.update)
		p.start()


		return self

	def info(title):
		print(title)
		print('module name:', __name__)
		print('parent process:', os.getppid())
		print('process id:', os.getpid())

	def update(self):
		cap = cv2.VideoCapture(self.src)
		#TimesQue=[]
		while self.started:
			s=datetime.datetime.now()
			grabbed, frame = cap.read()
			if not self.FramesQue.qsize()>=10:
				self.FramesQue.put(frame)
			else:
				self.FramesQue.get()
				self.FramesQue.put(frame)
			#e = (datetime.datetime.now()-s).total_seconds()
			#TimesQue.append(e)
			#if(len(TimesQue)==10):
			#	print(statistics.mean(TimesQue))
			#	print(statistics.stdev(TimesQue))
		cap.release()
	
	def read(self):

		if (not self.FramesQue.empty()):
			frame=self.FramesQue.get()
			grabbed=True

		else:
			startTime = datetime.datetime.now()
			maxWaitPeriod=1000
			while(self.FramesQue.empty()):
				# wait for some period
				time.sleep(maxWaitPeriod*0.001*0.1)
				deltaT=datetime.datetime.now()-startTime
				if (deltaT>datetime.timedelta(milliseconds=maxWaitPeriod)):

					frame = None
					grabbed = False
					print("Que was empty for more than "+ str(maxWaitPeriod)+" ms")
					return grabbed, frame

			frame = self.FramesQue.get()
			grabbed = True

		return grabbed,frame



	def stop(self):
		self.started = False

	def quelen(self):
		return self.FramesQue.qsize()

	def __exit__(self, exec_type, exc_value, traceback):
		self.started=False
	