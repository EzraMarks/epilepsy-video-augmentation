import cv2
import numpy as np
from queue import Queue
from threading import Thread
from progress.bar import IncrementalBar

from flash_detection import detect_flashes
exec(open("./video_augmentation.py").read())

class ProgressBar(IncrementalBar):
    suffix = "%(percent)d%% [%(elapsed_td)s / %(eta_td)s]"


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
        num_frames = 10
        overlap = 0
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
                frames[:, :, :, num_frames - overlap +
                       i] = self.input_queue.queue[i]

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

        while not self.input_queue.empty():
            bar.next()
            self.output_queue.put(self.input_queue.get())

        bar.finish()
        if total > 0:
            print((total_after / total) * 100)
        print(total)


class WritingThread():
    def __init__(self, output_queue, original_queue):
        self.name = "Writing Thread"
        self.output_queue = output_queue
        self.original_queue = original_queue
        self.run()

    def run(self):
        while self.original_queue.qsize() > 0:
            # read altered frame from queue
            frame = self.output_queue.get()
            # read original frame from queue
            oframe = self.original_queue.get()
            # concatenate them together with the original frame on the left and
            # altered frame on the right
            disp = np.concatenate((oframe, frame), axis=1)
            cv2.imshow('frame', disp)
            cv2.waitKey(waitFor)
        cv2.destroyAllWindows()


def detect_flashes(frames):
    resolution = 5
    num_frames = frames.shape[3]
    # height and width of an image region in which to detect total brightness change
    window_h = int(frames.shape[0] / resolution)
    window_w = int(frames.shape[1] / resolution)

    # luminance change from one frame to the next
    lumen_changes = np.zeros(
        (resolution, resolution, num_frames - 1), dtype=np.int32)

    # fill array of luminance changes
    for idx in range(num_frames - 1):
        f_curr = frames[:, :, :, idx].astype(np.int16)
        f_next = frames[:, :, :, idx + 1].astype(np.int16)
        f_diff = f_next - f_curr

        for i in range(resolution):
            for j in range(resolution):
                lumen_changes[i, j, idx] = np.sum(f_diff[window_h * i: window_h * (i + 1),
                                                         window_w * j: window_w * (j + 1)])

    abs_lumen_changes = np.abs(lumen_changes)
    # threshold for how much total lumenence variation constitutes a flash
    threshold = 51 * num_frames * window_h * window_w  # TODO dial in threshold

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

# create video reader
video = cv2.VideoCapture("../source-footage/shock.mp4")

if not video.isOpened():
    print("Error Opening Video File")
waitFor = int(1000.0 / video.get(cv2.CAP_PROP_FPS))
frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create instances of each thread
reading_thread = ReadingThread(input_queue, video, original_queue)
processing_thread = ProcessingThread(
    input_queue, output_queue, frame_w, frame_h)
writing_thread = WritingThread(output_queue, original_queue)

print('Video finished playing')
