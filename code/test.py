import cv2
import numpy as np

def main():
    video = cv2.VideoCapture("video.mp4")

    if not video.isOpened():
        print("Error Opening Video File")

    ret, frame = video.read()
    cv2.imshow('frame', frame)
    video_shape = frame.shape
    buffer_size = 100

    frame_buffer = np.zeros((video_shape[0],
                            video_shape[1],
                            video_shape[2],
                            buffer_size))

    # TODO proram currently crashes when ret = false;
    # make it terminate gracefully, e.g.
    # if (not ret): break
    while video.isOpened():
        for i in range(buffer_size):
            ret, frame = video.read()
            # if ret:
                # cv2.imshow('frame', frame)
            # frame_buffer[:, :, :, i] = frame
        #     print(frame_buffer[:, :, :, i].shape)
        #     print(frame.shape)

        #     if ret:
        #         cv2.imshow('frame', frame)
            
        
        # detect_flashes(frame_buffer)

        # for i in range(buffer_size):
        #     cv2.imshow('frame', frame_buffer[:, :, :, i])

    video.release()
    cv2.destroyAllWindows()

def detect_flashes(frame_buffer):
    print("cool")


#     f_diff = f_

# Y (cd/m2) = 413.435(0.002745 * Yd + 0.0189623) ** 2.2

main()