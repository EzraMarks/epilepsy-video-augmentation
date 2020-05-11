from queue import Queue
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from cv2 import cvtColor
from progress.bar import IncrementalBar

class ProgressBar(IncrementalBar):
    suffix= "%(percent)d%% [%(elapsed_td)s / %(eta_td)s]"


class ReadingThread():
    def __init__(self, input_queue, video, original_queue):
        self.name = "Reading Thread"
        self.input_queue = input_queue
        self.video = video
        self.original_queue = original_queue
        self.run()

    def run(self):
        bar = ProgressBar("Reading", max=video.get(cv2.CAP_PROP_FRAME_COUNT))
        go = self.video.isOpened()
        while (go):
            # load frames into queue
            go, frame = self.video.read()
            if (go):
                self.input_queue.put(frame)
                self.original_queue.put(frame)
            bar.next()
        bar.finish()
        self.video.release()


class ProcessingThread():
    def __init__(self, input_queue, output_queue, frame_w, frame_h):
        self.name = "Processing Thread"
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.run()

    def run(self):
        bar = ProgressBar("Processing", max=input_queue.qsize())
        num_frames = 12
        overlap = 2
        frames = np.zeros((frame_h, frame_w, 3, num_frames), dtype=np.uint8)
        total_after = 0
        total = 0
        while (self.input_queue.qsize() >= num_frames):
            # loading frames into array for processing:
            for i in range(num_frames - overlap):
                bar.next()
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
                frames = blend(frames)
                total = total + 1
                if detect_flashes(frames):
                    total_after = total_after + 1
            for i in range(num_frames - overlap):
                frame = np.copy(frames[:, :, :, i])
                self.output_queue.put(frame)
                ## this implementation attempts to slow down the video
                ## every time there are detected flashes. works, but it
                ## is pretty annoying since it slows everything down.
                # if flash:
                #     self.output_queue.put(frame)

        while not self.input_queue.empty():
            bar.next()
            self.output_queue.put(self.input_queue.get())

        bar.finish()
        if total > 0:
            print((total_after / total) * 100)
        print(total)

def normalize_brightness(frames):
    # some random arbitrary threshold I set
    threshold = 50
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)
    cv2.imshow('orig_frame', frames[:, :, :, 5])
    cv2.waitKey(5000)
    # calculate sum total value of all frames (in hsv, value corresponds to brightness SUPPOSEDLY)
    # in the real world this feels like a lie
    value_sum = np.zeros((frames.shape[0], frames.shape[1]))
    for i in range(num_frames):
        # display each original frame (without modification) for ease
        # cv2.imshow('orig_frame', frames[:, :, :, i])
        # cv2.waitKey(5)
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
        # cv2.imshow('new_frame', rgb_frame)
        # cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    # cv2.imshow('copied_frame', frames_cpy[:, :, :, 5])
    # cv2.waitKey(5000)
    return frames_cpy

# "absolute most lazy implementation"
# • This function replaces the first half of the frames with the first frame
# and the second half of the frames with the last frame.
# • It works pretty well on hand animation like pokemon shock, but it looks
# super jank on the seven nation army video since it
def lazy_stuff(frames):
    num_frames = frames.shape[3]
    over_two = int(num_frames / 2)
    # replace all the frames in the first half with the first frame
    for i in range(0, over_two):
        frames[:, :, :, i] = frames[:, :, :, 0]
    # replace all the frames in the second half with the last frame
    for i in range(over_two, num_frames):
        frames[:, :, :, i] = frames[:, :, :, num_frames - 1]
    return frames

# "the powerpoint transition"
# • This function takes the first frame and the last frame and progressively
# fades from the first one to the last one in num_frames steps, replacing
# the flashy frames with the fade.
# • Basically, it's the powerpoint transition we all know and love from
# middle school teachers that got too excited trying to make their
# presentations fun.
# • Looks okay on animation like pokemon shock, looks janky on seven nation
# army but slightly better than lazy_stuff.
def blend(frames):
    num_frames = frames.shape[3]
    inc = 1 / (num_frames - 2)
    beta = inc
    alpha = (1.0 - beta)
    for i in range(1, num_frames - 1):
        frames[:, :, :, i] = cv2.addWeighted(frames[:, :, :, 0], alpha, frames[:, :, :, num_frames - 1], beta, 0.0)
        beta = beta + inc
        alpha = 1.0 - beta
    # cv2.imshow('frame1', frames[:,:,:,0])
    # cv2.waitKey(5000)
    # cv2.imshow('frame2', frames[:,:,:,5])
    # cv2.waitKey(5000)
    # cv2.imshow('frame3', frames[:,:,:,11])
    # cv2.waitKey(5000)
    return frames

# "what if..."
# • this literally just combines blend and lazy_stuff (making it blazy boi)
# • I think blend looks better, it just becomes a bit too flashy still
def blazy_boi(frames):
    num_frames = frames.shape[3]
    over_two = int(num_frames / 2)
    begin = frames[:, :, :, 0]
    middle = frames[:, :, :, over_two - 1]
    end = frames[:, :, :, num_frames - 1]
    inc = 1 / (over_two - 2)
    beta = inc
    alpha = (1.0 - beta)
    for i in range(1, over_two - 1):
        frames[:, :, :, i] = cv2.addWeighted(begin, alpha, middle, beta, 0.0)
        beta = beta + inc
        alpha = 1.0 - beta
    beta = inc
    alpha = (1.0 - beta)
    for i in range(over_two, num_frames):
        frames[:, :, :, i] = cv2.addWeighted(middle, alpha, end, beta, 0.0)
        beta = beta + inc
        alpha = 1.0 - beta
    return frames

# "trying legit methods"
# • drops the contrast of the video and bumps brightness to offset
# • definitely gets rid of a lot of flashing I think but it looks very obviously
# altered, as it just gets really grey
def contrast_drop(frames):
    num_frames = frames.shape[3]
    # cv2.imshow('frame2', frames[:,:,:,11])
    # cv2.waitKey(5000)
    for i in range(0, num_frames):
        frames[:,:,:,i] = cv2.convertScaleAbs(frames[:,:,:,i], alpha=0.2, beta=100.0)
    # cv2.imshow('frame3', frames[:,:,:,11])
    # cv2.waitKey(5000)
    return frames


def blazy_contrast(frames):
    num_frames = frames.shape[3]
    over_two = int(num_frames / 2)
    begin = frames[:, :, :, 0]
    middle = cv2.convertScaleAbs(frames[:,:,:,over_two - 1], alpha=0.2, beta=100.0)
    end = frames[:, :, :, num_frames - 1]
    inc = 1 / num_frames
    alpha = (over_two - 1) * inc
    beta = 1.0 - alpha
    for i in range(1, over_two - 1):
        frames[:, :, :, i] = cv2.addWeighted(begin, alpha, middle, beta, 0.0)
        alpha = alpha - inc
        beta = 1.0 - alpha
    alpha = (num_frames + 1 - over_two) * inc
    beta = 1.0 - alpha
    for i in range(over_two, num_frames):
        frames[:, :, :, i] = cv2.addWeighted(middle, alpha, end, beta, 0.0)
        alpha = alpha - inc
        beta = 1.0 - alpha
    frames[:,:,:,over_two - 1] = middle
    return frames

def black_out(frames):
    return np.zeros((frame_h, frame_w, 3, frames.shape[3]), dtype=np.uint8)

def normalize_luminance(frames):
    num_frames = frames.shape[3]
    for i in range(0, num_frames):
        img = frames[:,:,:,i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV);
        channels = cv2.split(img);
        channels[0] = cv2.equalizeHist(channels[0]);
        img = cv2.merge(channels);
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB);
    return frames

class WritingThread():
    def __init__(self, output_queue, original_queue):
        self.name = "Writing Thread"
        self.output_queue = output_queue
        self.original_queue = original_queue
        self.run()

    def run(self):
        print(frame_w)
        print(frame_h)
        # out = cv2.VideoWriter()
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # if not out.open('outpy.mp4',fourcc, video.get(cv2.CAP_PROP_FPS), (2*frame_w,frame_h)):
        #     print("ruh rih")
        # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), video.get(cv2.CAP_PROP_FPS), (frame_w*2,frame_h))

        # cv2.imshow('frame', np.zeros((frame_h, frame_w*2, 3), dtype=np.uint8))
        # cv2.waitKey(10000)

        while self.original_queue.qsize() > 0: # TODO close window when video is over
            # read frames from queue
            frame = self.output_queue.get()
            oframe = self.original_queue.get()
            disp = np.concatenate((oframe,frame),axis=1)
            # out.write(disp)
            cv2.imshow('frame', disp)
            cv2.waitKey(waitFor)
        # out.release()
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

# FIFO (first-in-first-out) queue to hold frames after reading them in
input_queue = Queue()
# FIFO queue to hold frames after processing, before writing/displaying them out
output_queue = Queue()

original_queue = Queue()

obscure = "average_brightness"

# create video reader
video = cv2.VideoCapture("sherlock.mp4")
# video = cv2.VideoCapture("../results/shock_" + obscure + "_solo.mp4")

if not video.isOpened():
    print("Error Opening Video File")
waitFor = int(1000.0 / video.get(cv2.CAP_PROP_FPS))
frame_w  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create instances of each thread
reading_thread = ReadingThread(input_queue, video, original_queue)
processing_thread = ProcessingThread(input_queue, output_queue, frame_w, frame_h)
writing_thread = WritingThread(output_queue, original_queue)

print('Main Terminating...')
