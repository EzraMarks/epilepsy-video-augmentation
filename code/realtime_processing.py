import cv2
import numpy as np
from queue import Queue
from threading import Thread

from flash_detection import detect_flashes

class ReadingThread(Thread):
    def __init__(self, input_queue, video):
        Thread.__init__(self)
        self.name = "Reading Thread"
        self.input_queue = input_queue
        self.video = video
    
    def run(self):
        while (self.video.isOpened()):
            # load frames into queue
            ret, frame = self.video.read()
            if (ret):
                self.input_queue.put(frame)
            else:
                self.video.release()

class ProcessingThread(Thread):
    def __init__(self, input_queue, output_queue, video, augmentation_func):
        Thread.__init__(self)
        self.name = "Processing Thread"
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.video = video
        self.augmentation_func = augmentation_func
    
    def run(self):
        frame_w  = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = 10
        overlap = 0
        frames = np.zeros((frame_h, frame_w, 3, num_frames), dtype=np.uint8)

        while ((self.video.isOpened()) or (self.input_queue.qsize() >= num_frames)):
            if (self.input_queue.qsize() >= num_frames):
                # loading frames into array for processing:
                for i in range(num_frames - overlap):
                    # pop these frames off of the input queue,
                    # since we will no longer need them
                    frames[:, :, :, i] = self.input_queue.get()
                for i in range(overlap):
                    # keep these frames in the input queue for reuse (overlap)
                    # when processing the next segment of the video
                    frames[:, :, :, num_frames - overlap + i] = self.input_queue.queue[i]
                
                flash = detect_flashes(frames)

                if flash:    
                    frames = self.augmentation_func(frames)
                for i in range(num_frames - overlap):
                    frame = np.copy(frames[:, :, :, i])
                    self.output_queue.put(frame)

class WritingThread(Thread):
    def __init__(self, output_queue, video):
        Thread.__init__(self)
        self.name = "Writing Thread"
        self.output_queue = output_queue
        self.video = video
    
    def run(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        while ((self.video.isOpened()) or (self.output_queue.qsize() > 0)):
            # read frames from queue
            frame = self.output_queue.get()
            cv2.imshow('frame', frame)
            cv2.waitKey(int(1000 / fps))
        
        cv2.destroyAllWindows()