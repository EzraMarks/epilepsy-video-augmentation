## Usage:
Run main.py, passing in a video file as a command line argument:
```
python3 main.py my_video.mp4
```

Optionally, specify the type of video augmentation with the `--augmentation` flag:
```
python3 main.py my_video.mp4 --augmentation contrast_drop
```

For a full list of video augmentation options, use the `--help` flag:
```
python3 main.py --help
```

If using an OS that does not support multithreading, use the `--preprocess` flag. This will preprocess the video, rather than using multithreading for real-time video processing and viewing.
