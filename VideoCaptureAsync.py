import threading
import cv2
import queue

class VideoCaptureAsync:
	def __init__(self, src=0):
		self.src = src
		self.cap = cv2.VideoCapture(self.src)
		self.grabbed, self.frame = self.cap.read()
		self.started = False
		self.FramesQue=queue.Queue(10)
		#self.read_lock = threading.Lock()

	def set(self, var1, var2):
		self.cap.set(var1, var2)
	
	def start(self):
		if self.started:
			print('[!] Asynchroneous video capturing has already been started.')
			return None
		self.started = True
		self.thread = threading.Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()
		return self
	
	def update(self):
		while self.started:
			grabbed, frame = self.cap.read()
			#with self.read_lock:
			self.grabbed = grabbed
			self.frame = frame
			if not self.FramesQue.full():
				self.FramesQue.put(frame)
			else:
				self.FramesQue.get()
				self.FramesQue.put(frame)
	
	def read(self):
		#with self.read_lock:
		if (not self.FramesQue.empty()):
			frame=self.FramesQue.get()
			grabbed=True
		else:
			frame=None
			grabbed=False
		return grabbed, frame
	
	def stop(self):
		self.started = False
		self.thread.join()
	def quelen(self):
		return self.FramesQue.qsize()
	def __exit__(self, exec_type, exc_value, traceback):
		self.cap.release()
	