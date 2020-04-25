from threading import Thread
from queue import Queue
import numpy as np
import cv2
import time

class ReadingThread(Thread):
    def __init__(self, input_queue, filename):
        Thread.__init__(self)
        self.name = "Reading Thread"
        self.input_queue = input_queue
        self.filename = filename
    
    def run(self):
        while video.isOpened():
            # load frames into queue
            ret, frame = video.read()
            if(ret):
                self.input_queue.put(frame)

        video.release()


class ProcessingThread(Thread):
    def __init__(self, input_queue, output_queue, frame_w, frame_h):
        Thread.__init__(self)
        self.name = "Processing Thread"
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.frame_w = frame_w
        self.frame_h = frame_h
    
    def run(self):
        num_frames = 12
        overlap = 2
        frames = np.zeros((frame_h, frame_w, 3, num_frames), dtype=np.uint8)

        while (True):
            if (self.input_queue.qsize() >= num_frames):
                # load frames into array -- pop these frames off of the input queue,
                # since we will no longer need them
                for i in range(num_frames - overlap):
                    frames[:, :, :, i] = self.input_queue.get()
                # load remaining frames into array -- keep them in the input queue
                # for reuse (overlap) when processing the next segment of the video
                for i in range(overlap):
                    frames[:, :, :, num_frames - overlap + i] = self.input_queue.queue[i]
                
                flash = detect_flashes(frames)
                print(flash)

                for i in range(num_frames - overlap):
                    frame = np.copy(frames[:, :, :, i])
                    if flash:
                        frame = frame // 10
                    self.output_queue.put(frame)
                    


def detect_flashes(frames):
    resolution = 5
    num_frames = frames.shape[3]
    # height and width of an image region in which to detect total brightness change
    window_h = int(frames.shape[0] / resolution)
    window_w = int(frames.shape[1] / resolution)


    # luminance change from one frame to the next
    lumen_changes = np.zeros((resolution, resolution, num_frames - 1), dtype=np.int32)

    # fill array of luminance_changes
    # TODO not currently accounting for the final 1 frame in the buffer
    for idx in range(num_frames - 1):
        f_curr = frames[:, :, :, idx].astype(np.int16)
        f_next = frames[:, :, :, idx + 1].astype(np.int16)
        f_diff = f_next - f_curr

        for i in range(resolution):
            for j in range(resolution):
                lumen_changes[i, j, idx] = np.sum(f_diff[window_h * i : window_h * (i + 1),
                                                                window_w * j : window_w * (j + 1)])

    abs_lumen_changes = np.abs(lumen_changes)
    # threshold for how much total lumenence variation constitutes a flash
    threshold = 51 * num_frames * window_h * window_w # TODO dial in threshold

    # sum the amount that the brightness changes between all the
    # frames in this video segment -- if the brightness changes a lot,
    # there are probably flashes during this segment ¯\_(ツ)_/¯
    total_lumen_changes = np.sum(abs_lumen_changes, axis=2, dtype=np.int32)

    regional_flashes = total_lumen_changes > threshold

    # collapse regional flashes into single flash detection boolean
    flash = np.sum(regional_flashes) != 0

    return flash


class WritingThread(Thread):
    def __init__(self, output_queue):
        Thread.__init__(self)
        self.name = "Writing Thread"
        self.output_queue = output_queue
    
    def run(self):
        while (True):
            # read frames from queue
            print(self.output_queue.qsize())
            frame = self.output_queue.get()
            cv2.imshow('frame', frame)
            cv2.waitKey(33)
        
        cv2.destroyAllWindows()


# FIFO (first-in-first-out) queue to hold frames after reading them in
input_queue = Queue()
# FIFO queue to hold frames after processing, before writing/displaying them out
output_queue = Queue()

# create video reader
video = cv2.VideoCapture("video720p.mp4")
if not video.isOpened():
    print("Error Opening Video File")

frame_w  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create instances of each thread
reading_thread = ReadingThread(input_queue, video)
processing_thread = ProcessingThread(input_queue, output_queue, frame_w, frame_h)
writing_thread = WritingThread(output_queue)

# start running all threads
reading_thread.start()
processing_thread.start()
writing_thread.start()

# after all threads are finished
reading_thread.join()
processing_thread.join()
writing_thread.join()
print('Main Terminating...')