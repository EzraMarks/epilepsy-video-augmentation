# Video Augmentation for Photosensitive Epilepsy 

Photosensitivity, formally known as photosensitive epilepsy, is a condition that results in sensitivity to certain visual patterns, particularly flashing lights. When exposed to these visual stimuli, a person with photosensitivity can experience seizures or seizure-like symptoms. The Epilepsy Foundation has identified key triggers for light-induced seizures, including: the frequency of the flickering light; the intensity and contrast between the light and dark images that compose the flash; the total visual area occupied by the light stimulus; and the pattern of the image, particularly light and dark stripes.<sup>1</sup>

This program detects patterns of flashing light in video and outputs an augmented video with the photosensitive-triggering light patterns removed, achieved in near real-time. Although video producers are the first line of defense in removing content that could trigger photosensitivity, online video platforms like YouTube currently lack the necesary regulation. This software is meant directly for the end-user, giving agency and peace of mind for video viewers with photosensitivity.

Although this program has shown promising results on augmenting dangerous flashing videos like the “Pokémon Shock”<sup>2</sup> episode, the project is far from thoroughly tested. Please use at your own risk.

<sub>1 &nbsp; Epilepsy Foundation. “Shedding Light on Photosensitivity, One of Epilepsy's Most Complex Conditions.” Epilepsy Foundation, www.epilepsy.com/article/2014/3/shedding-light-photosensitivity-one-epilepsys-most-complex-conditions-0.</sup>

<sub> 2 &nbsp; When the 1997 episode of Pokémon (termed “Pokémon Shock”) debued, 685 children in Japan were hospitalized due to one scene with flashing red and blue lights. For more information, see “The Pokémon Panic of 1997” by Skeptical Inquirer (May 2001), https://web.archive.org/web/20020125093204/http://www.csicop.org/si/2001-05/pokemon.html.</sub>

## Usage:
To view a video with photosensitive triggers reduced in real-time, run main.py, passing in the video file as a command line argument:
```
python3 main.py my_video.mp4
```

If using an OS that does not support multithreading, use the `--preprocess` flag. This will preprocess the video, rather than using multithreading for real-time video processing and viewing:
```
python3 main.py my_video.mp4 --preprocess
```

Optionally, specify the type of video augmentation with the `--augmentation` flag:
```
python3 main.py my_video.mp4 --augmentation contrast_drop
```

For a full list of video augmentation options, use the `--help` flag:
```
python3 main.py --help
```
