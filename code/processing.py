import cv2
import numpy as np
from queue import Queue
from threading import Thread
from progress.bar import IncrementalBar

from flash_detection import detect_flashes

class ProgressBar(IncrementalBar):
    suffix = "%(percent)d%% [%(elapsed_td)s / %(eta_td)s]"

class ReadingThread(Thread):
    def __init__(self, input_queue, video):
        Thread.__init__(self)
        self.name = "Reading Thread"
        self.input_queue = input_queue
        self.video = video
    
    def run(self):
        isReading = self.video.isOpened()
        while (isReading):
            if (self.input_queue.qsize() < 480):
                # loads frames into queue
                isReading, frame = self.video.read()
                if (isReading):
                    self.input_queue.put(frame)
            else:
                cv2.waitKey(1000)

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
        progressBar = ProgressBar("Processing", max=self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w  = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = 10
        overlap = 0
        frames = np.zeros((frame_h, frame_w, 3, num_frames), dtype=np.uint8)

        while ((self.video.isOpened()) or (self.input_queue.qsize() >= num_frames)):
            if (self.input_queue.qsize() >= num_frames):
                # loads frames into array for processing
                for i in range(num_frames - overlap):
                    progressBar.next()
                    # pops used frames off of the input queue
                    frames[:, :, :, i] = self.input_queue.get()
                for i in range(overlap):
                    # peeks at overlapping frames in input queue, not removing them so
                    # that they can be reused when processing the next segment of the video
                    frames[:, :, :, num_frames - overlap + i] = self.input_queue.queue[i]
                
                flash = detect_flashes(frames)

                if flash:    
                    frames = self.augmentation_func(frames)
                for i in range(num_frames - overlap):
                    frame = np.copy(frames[:, :, :, i])
                    self.output_queue.put(frame)

        progressBar.finish()

class WritingThread(Thread):
    def __init__(self, input_queue, output_queue, video, video_writer = None):
        Thread.__init__(self)
        self.name = "Writing Thread"
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.video = video
        self.video_writer = video_writer
    
    def run(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        while self.video.isOpened() or (self.input_queue.qsize() > 10) or (self.output_queue.qsize() > 0):
            # pauses to buffer, if running too slowly
            if (self.output_queue.qsize() == 0):
                cv2.waitKey(2000)
                continue
            
            frame = self.output_queue.get()

            # live video playback
            if (self.video_writer == None):
                cv2.imshow('frame', frame)
                cv2.waitKey(int(1000 / fps))
            # writing video to file
            else:
                self.video_writer.write(frame)
        
        # closes video writing / video playback
        if (self.video_writer != None): self.video_writer.release()
        cv2.destroyAllWindows()