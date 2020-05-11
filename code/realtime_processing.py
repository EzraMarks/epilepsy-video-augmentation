import cv2
import numpy as np
from queue import Queue
from threading import Thread

exec(open("./video_augmentation.py").read())
exec(open("./flash_detection.py").read())


class ReadingThread(Thread):
    def __init__(self, input_queue, video):
        Thread.__init__(self)
        self.name = "Reading Thread"
        self.input_queue = input_queue
        self.video = video
    
    def run(self):
        while (video.isOpened()):
            # load frames into queue
            ret, frame = video.read()
            if (ret):
                input_queue.put(frame)
            else:
                video.release()


class ProcessingThread(Thread):
    def __init__(self, input_queue, output_queue, video):
        Thread.__init__(self)
        self.name = "Processing Thread"
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.video = video
    
    def run(self):
        frame_w  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = 10
        overlap = 0
        frames = np.zeros((frame_h, frame_w, 3, num_frames), dtype=np.uint8)

        while ((video.isOpened()) or (input_queue.qsize() >= num_frames)):
            if (input_queue.qsize() >= num_frames):
                # loading frames into array for processing:
                for i in range(num_frames - overlap):
                    # pop these frames off of the input queue,
                    # since we will no longer need them
                    frames[:, :, :, i] = input_queue.get()
                for i in range(overlap):
                    # keep these frames in the input queue for reuse (overlap)
                    # when processing the next segment of the video
                    frames[:, :, :, num_frames - overlap + i] = input_queue.queue[i]
                
                flash = detect_flashes(frames)

                if flash:    
                    frames = contrast_drop(frames) # TODO select video augmentation
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
        while ((video.isOpened()) or (output_queue.qsize() > 0)):
            # read frames from queue
            frame = self.output_queue.get()
            cv2.imshow('frame', frame)
            cv2.waitKey(33)
        
        cv2.destroyAllWindows()


# FIFO (first-in-first-out) queue to hold frames after reading them in
input_queue = Queue()
# FIFO queue to hold frames after processing, before writing/displaying them out
output_queue = Queue()

# create video reader
video = cv2.VideoCapture("../source-footage/shock.mp4")
if not video.isOpened():
    print("Error Opening Video File")

# create instances of each thread
reading_thread = ReadingThread(input_queue, video)
processing_thread = ProcessingThread(input_queue, output_queue, video)
writing_thread = WritingThread(output_queue, video)

# start running all threads
reading_thread.start()
processing_thread.start()
writing_thread.start()

# after all threads are finished
reading_thread.join()
processing_thread.join()
writing_thread.join()
print('Video finished playing')
