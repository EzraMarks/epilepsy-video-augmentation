import cv2
import numpy as np


BUFFER_SIZE = 300
RESOLUTION = 5

def main():

    video = cv2.VideoCapture("video3.mp4")

    if not video.isOpened():
        print("Error Opening Video File")

    ret, frame = video.read()
    video_shape = frame.shape

    frame_buffer = np.empty((video_shape[0],
                            video_shape[1],
                            video_shape[2],
                            BUFFER_SIZE))

    while video.isOpened():
        # load frames into buffer
        for i in range(BUFFER_SIZE):
            ret, frame = video.read()
            frame_buffer[:, :, :, i] = frame / 255
            
        flashes = detect_flashes(frame_buffer)

        # read frames from buffer
        for i in range(BUFFER_SIZE):
            frame = frame_buffer[:, :, :, i]
            print(flashes[i])
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


def detect_flashes(frame_buffer):
    # height and width of an image region in which to detect total brightness change
    window_h = int(frame_buffer.shape[0] / RESOLUTION)
    window_w = int(frame_buffer.shape[1] / RESOLUTION)

    # luminance change from one frame to the next
    luminance_changes = np.zeros((RESOLUTION, RESOLUTION, BUFFER_SIZE))
    # array of booleans (1 if flash detected, 0 otherwise)
    regional_flashes = np.zeros((RESOLUTION, RESOLUTION, BUFFER_SIZE))

    # fill array of luminance_changes
    # TODO not currently accounting for the final 1 frame in the buffer
    for idx in range(BUFFER_SIZE - 1):
        f_curr = frame_buffer[:, :, :, idx]
        f_next = frame_buffer[:, :, :, idx + 1]
        f_diff = f_next - f_curr

        for i in range(RESOLUTION):
            for j in range(RESOLUTION):
                luminance_changes[i, j, idx] = np.sum(f_diff[window_h * i : window_h * (i + 1),
                                                               window_w * j : window_w * (j + 1)])
    
    abs_lumen_changes = np.abs(luminance_changes)
    # threshold for how much total lumenence variation constitutes a flash
    threshold = 0.2 * 10 * (window_h * window_w) # TODO dial in threshold

    # fill in the array of detected flashes:
    # the methodology here is generally to sum the amount that the brightness
    # changes during the duration of 10 frames -- if the brightness changes a lot,
    # then there are probably flashes during those 10 frames ¯\_(ツ)_/¯
    # TODO not currently accounting for the final 10 frames in the buffer
    for idx in range(0, BUFFER_SIZE - 10, 10):
        for i in range(RESOLUTION):
            for j in range(RESOLUTION):
                lumen_change = np.sum(abs_lumen_changes[i, j, idx : idx + 10])

                if (lumen_change > threshold):
                    regional_flashes[i, j, idx : idx + 10] = 1

    # collapse regional flashes into simple array of yes/no flash in a frame
    flashes = np.sum(np.sum(regional_flashes, axis=0), axis=0)
    flashes[flashes != 0] = 1

    return flashes

main()