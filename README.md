# noisy-speech-data-augmentor

## Description

CLI tool to take speech audio input and add noise to augment noisy speech data for training machine learning models on limited data

## How to Run

### Installing Dependencies

This tool was written using Python 3.9.6, and the following external libraries:
- numpy
- soundfile
- glob
- scipy

Each of these dependencies can be installed using `pip3 install <library-name>` or `pip install <library-name>`depending on your configurations

### Running the program

The program can simply be run using `python3 noise.py`. Desired noise must be specified using `-a`, `-m`, `-r`, `-c`, or `-d`. Input directory containing audio files can be specified by using `-i`. If not set, the program defaults to `samples/SpeechSamples`. The desired output directory can also be specified using the `-o` command and subdirectories will be created for each of the desired forms of noise. Default is `outputs`.

I/O arguments:
`-i`: Optional. Specifies input directory for input audio speech files. Defaults to `samples/SpeechSamples`

`-o`: Optional. Specifies output directory. Creates `/output` by default.

`-n`: Specifies directory for ambient noise samples to be added to input. Is not included if omitted or invalid.

Noise arguments:
`-a`: Adds mechanical whirr, next room, cutout, and delay effects, and combines all effects. Output files are stored in `<specifiedFolder>/mixed`

`-m`: Adds mechanical whirr to mimic room fan noises. Output files are stored in `<specifiedFolder>/mech_whirr`

`-r`: Adds muffled effect to mimic input audio coming from another room. Output files are stored in `<specifiedFolder>/room`

`-c`: Adds random cuts to the input audio to simulate packet drops or bad phone connection. Output files are stored in `<specifiedFolder>/cutout`

`-d`: Adds delay/echo effect to input audio to simulate feedback or room echo. Output files are stored in `<specifiedFolder>/delay`

## Ambient noise
Ambient audio files such as the ones available on kaggle (ie. https://www.kaggle.com/datasets/nafin59/hospital-ambient-noise?resource=download) are recommended for the noise files.

## Mentions
Audio filters for low-pass and band-pass filters were modified/adadpted from code available from https://thewolfsound.com/