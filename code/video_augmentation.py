import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.color import rgb2hsv

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
        frames[:, :, :, i] = cv2.addWeighted(
            frames[:, :, :, 0], alpha, frames[:, :, :, num_frames - 1], beta, 0.0)
        beta = beta + inc
        alpha = 1.0 - beta
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
    for i in range(0, num_frames):
        frames[:, :, :, i] = cv2.convertScaleAbs(
            frames[:, :, :, i], alpha=0.2, beta=100.0)
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

def blazy_contrast(frames):
    num_frames = frames.shape[3]
    over_two = int(num_frames / 2)
    begin = frames[:, :, :, 0]
    middle = cv2.convertScaleAbs(
        frames[:, :, :, over_two - 1], alpha=0.2, beta=100.0)
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
    frames[:, :, :, over_two - 1] = middle
    return frames

def average_brightness(frames):
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # calculate sum total value of all frames (in hsv, value corresponds to brightness SUPPOSEDLY)
    # in the real world this feels like a lie
    value_sum = np.zeros((frames.shape[0], frames.shape[1]))
    for i in range(num_frames):
        # convert to HSV space to extract values
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        value = hsv_frame[:, :, 2]
        # add value to running sum (avg later)
        value_sum += value

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

        # convert frame back to RGB
        rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy

def threshold_brightness(frames):
    # some random arbitrary threshold I set
    threshold = 50
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # average over entire images (could be use to normalize, ie subtract out)
    rgb_sum = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
    for i in range(num_frames):
        rgb_sum += frames[:, :, :, i]
    rgb_avg = rgb_sum/num_frames

    # just take the average of like every single pixel
    sum_avg = np.sum(rgb_avg)
    total_avg = sum_avg/(frames.shape[0] * frames.shape[1])


    for j in range(num_frames):
        # get each frame
        rgb_frame = frames[:, :, :, j]

        # threshold based on average pixel value in order to make outliers less harsh (this didn't look *horrible*)
        # (just very bad) -> also tried thresholding brightness and that was worse
        rgb_frame_hthresh = rgb_frame > total_avg + threshold
        rgb_frame_lthresh = rgb_frame < total_avg - threshold
        rgb_frame[rgb_frame_hthresh] = rgb_frame[rgb_frame_hthresh] - threshold
        rgb_frame[rgb_frame_lthresh] = rgb_frame[rgb_frame_lthresh] + threshold

        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy

def normalize_brightness(frames):
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # calculate sum total value of all frames (in hsv, value corresponds to brightness SUPPOSEDLY)
    # in the real world this feels like a lie
    value_sum = np.zeros((frames.shape[0], frames.shape[1]))
    for i in range(num_frames):
        # convert to HSV space to extract values
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        value = hsv_frame[:, :, 2]
        # add value to running sum (avg later)
        value_sum += value

    for j in range(num_frames):
        # get each frame
        rgb_frame = frames[:, :, :, j]
        # convert from RGB to HSV space to fuck with value (brightness???)
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        # set brightness of every pixel to same value (starting to think this *isn't* brightness)
        hsv_frame[:, :, 2] = hsv_frame[:, :, 2] - value_sum

        # convert frame back to RGB
        rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy

def normalize_pixels(frames):
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # average over entire images (could be use to normalize, ie subtract out)
    rgb_sum = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
    for i in range(num_frames):
        rgb_sum += frames[:, :, :, i]
    rgb_avg = rgb_sum/num_frames


    for j in range(num_frames):
        # get each frame
        rgb_frame = frames[:, :, :, j] - rgb_avg
        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy

def average_lab(frames):
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # calculate sum total value of all frames (in hsv, value corresponds to brightness SUPPOSEDLY)
    # in the real world this feels like a lie
    lab_sum = np.zeros((frames.shape[0], frames.shape[1]))
    for i in range(num_frames):
        # display each original frame (without modification) for ease
        # convert to LAB space to extract luminance
        lab_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2LAB)
        lab = lab_frame[:, :, 0]
        # add value to running sum (avg later)
        lab_sum += lab

        # average brightness across all images as a single value
    avg_lab = np.sum(lab_sum/num_frames)/(frames.shape[0]*frames.shape[1])
    # try just making the brightness of every image the same (current attempt)
    brightness = np.full((frames.shape[0], frames.shape[1]), avg_lab)

    for j in range(num_frames):
        # get each frame
        rgb_frame = frames[:, :, :, j]
        # convert from RGB to HSV space to fuck with value (brightness???)
        lab_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        # set brightness of every pixel to same value (starting to think this *isn't* brightness)
        lab_frame[:, :, 0] = brightness

        # convert frame back to RGB
        rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_LAB2RGB)
        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy

def replace_value(frames):
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # calculate sum total value of all frames (in hsv, value corresponds to brightness SUPPOSEDLY)
    # in the real world this feels like a lie
    value_sum = np.zeros((frames.shape[0], frames.shape[1]))
    for i in range(num_frames):
        # convert to HSV space to extract values
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        value = hsv_frame[:, :, 2]
        # add value to running sum (avg later)
        value_sum += value

    # average brightness across all images as a single value
    avg_value = np.sum(value_sum/num_frames)/(frames.shape[0]*frames.shape[1])

    for j in range(num_frames):
        # get each frame
        rgb_frame = frames[:, :, :, j]
        # convert from RGB to HSV space to fuck with value (brightness???)
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2HSV)
        # set brightness of every pixel to same value (starting to think this *isn't* brightness)
        hsv_frame[:, :, 2] = avg_value

        # convert frame back to RGB
        rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy

def replace_luminance(frames):
    num_frames = frames.shape[3]
    # not copying the frames modifies all of them idk why
    frames_cpy = np.copy(frames)

    # calculate sum total value of all frames (in hsv, value corresponds to brightness SUPPOSEDLY)
    # in the real world this feels like a lie
    value_sum = np.zeros((frames.shape[0], frames.shape[1]))
    for i in range(num_frames):
        # convert to HSV space to extract values
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2LAB)
        value = hsv_frame[:, :, 2]
        # add value to running sum (avg later)
        value_sum += value

    # average brightness across all images as a single value
    avg_value = np.sum(value_sum/num_frames)/(frames.shape[0]*frames.shape[1])

    for j in range(num_frames):
        # get each frame
        rgb_frame = frames[:, :, :, j]
        # convert from RGB to HSV space to fuck with value (brightness???)
        hsv_frame = cv2.cvtColor(frames[:, :, :, i], cv2.COLOR_RGB2LAB)
        # set brightness of every pixel to same value (starting to think this *isn't* brightness)
        hsv_frame[:, :, 2] = avg_value

        # convert frame back to RGB
        rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_LAB2RGB)
        # display frame so as to better see my pain
        cv2.imshow('new_frame', rgb_frame)
        cv2.waitKey(5)
        # modify frame in copied array
        frames_cpy[:, :, :, j] = rgb_frame
    return frames_cpy