vidReader = VideoReader("video.mp4");

while hasFrame(vidReader)
    frameRGB = readFrame(vidReader);
    frameGrey = rgb2gray(frameRGB);
    imshow(frameGrey)
end