import cv2
import numpy as np

def main():

    video = cv2.VideoCapture("video.mp4")

    if not video.isOpened():
        print("Error Opening Video File")

    ret, frame = video.read()
    video_shape = frame.shape
    buffer_size = 100

    frame_buffer = np.empty((video_shape[0],
                            video_shape[1],
                            video_shape[2],
                            buffer_size))

    while video.isOpened():
        # load frames into buffer
        for i in range(buffer_size):
            ret, frame = video.read()
            frame_buffer[:, :, :, i] = frame / 255
            

        detect_flashes(frame_buffer)

        # read frames from buffer
        for i in range(buffer_size):
            frame = frame_buffer[:, :, :, i]
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


def detect_flashes(frame_buffer):
    for i in range((frame_buffer.shape[3])):
        frame_buffer[100:200, 100:200, :, i] = 0.0

# Y (cd/m2) = 413.435(0.002745 * Yd + 0.0189623) ** 2.2

main()