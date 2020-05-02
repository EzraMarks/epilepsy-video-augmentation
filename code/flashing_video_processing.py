from threading import Thread
from queue import Queue
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from cv2 import cvtColor


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

        self.video.release()


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
                # if a flash is detected, call extremely cursed function
                if flash:    
                    frames = normalize_brightness(frames)
                for i in range(num_frames - overlap):
                    frame = np.copy(frames[:, :, :, i])
                    self.output_queue.put(frame)


class WritingThread(Thread):
    def __init__(self, output_queue):
        Thread.__init__(self)
        self.name = "Writing Thread"
        self.output_queue = output_queue
    
    def run(self):
        while (True): # TODO close window when video is over
            # read frames from queue
            frame = self.output_queue.get()
            cv2.imshow('frame', frame)
            cv2.waitKey(33)
        
        cv2.destroyAllWindows()

def detect_flashes(frames):
    resolution = 5
    num_frames = frames.shape[3]
    # height and width of an image region in which to detect total brightness change
    window_h = int(frames.shape[0] / resolution)
    window_w = int(frames.shape[1] / resolution)

    # luminance change from one frame to the next
    lumen_changes = np.zeros((resolution, resolution, num_frames - 1), dtype=np.int32)

    # fill array of luminance changes
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
    # there are likely flashes during this segment
    total_lumen_changes = np.sum(abs_lumen_changes, axis=2, dtype=np.int32)

    regional_flashes = total_lumen_changes > threshold

    # collapse regional flashes into single flash detection boolean
    flash = np.sum(regional_flashes) != 0

    return flash

def normalize_brightness(frames):
    # some random arbitrary threshold I set 
    threshold = 50
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # calculate sum total value of all frames (in hsv, value corresponds to brightness SUPPOSEDLY)
    # in the real world this feels like a lie
    value_sum = np.zeros((frames.shape[0], frames.shape[1]))
    for i in range(num_frames):
        # display each original frame (without modification) for ease
        cv2.imshow('orig_frame', frames[:, :, :, i])
        cv2.waitKey(5)
        # convert to HSV space to extract values
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        value = hsv_frame[:, :, 2]
        # add value to running sum (avg later)
        value_sum += value  

    # average over entire images (could be use to normalize, ie subtract out)
    # rgb_sum = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
    # for i in range(num_frames):
    #     rgb_sum += frames[:, :, :, i]
    # rgb_avg = rgb_sum/num_frames

    # just take the average of like every single pixel
    # sum_avg = np.sum(rgb_avg)
    # total_avg = sum_avg/(frames.shape[0] * frames.shape[1])

    # average brightness across all images as a single value
    avg_value = np.sum(value_sum/num_frames)/(frames.shape[0]*frames.shape[1])
    # try just making the brightness of every image the same (current attempt)
    brightness = np.full((frames.shape[0], frames.shape[1]), avg_value)

    for j in range(num_frames):
        # get each frame
        rgb_frame = frames[:, :, :, j]
        # convert from RGB to HSV space to fuck with value (brightness???)
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        # set brightness of every pixel to same value (starting to think this *isn't* brightness)
        hsv_frame[:, :, 2] = brightness

        # threshold based on average pixel value in order to make outliers less harsh (this didn't look *horrible*)
        # (just very bad) -> also tried thresholding brightness and that was worse
        # rgb_frame_hthresh = rgb_frame > total_avg + threshold
        # rgb_frame_lthresh = rgb_frame < total_avg - threshold
        # rgb_frame[rgb_frame_hthresh] = rgb_frame[rgb_frame_hthresh] - threshold
        # rgb_frame[rgb_frame_lthresh] = rgb_frame[rgb_frame_lthresh] + threshold 

        # convert frame back to RGB
        rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy


# FIFO (first-in-first-out) queue to hold frames after reading them in
input_queue = Queue()
# FIFO queue to hold frames after processing, before writing/displaying them out
output_queue = Queue()

# create video reader
video = cv2.VideoCapture("video.mp4")
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
