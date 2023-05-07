import numpy as np
import soundfile as sf
import os
import sys
import random

from glob import glob

import librosa
import librosa.display

# new_audio_data = audio_data
# add_audio_data, _ = librosa.load(audio_files[1])

# for i in range(len(new_audio_data)):
#     new_audio_data[i] = new_audio_data[i] + add_audio_data[i]

# sf.write("outputAudio.wav", new_audio_data, sr)

# path = os.path.join(".", "outputs")
# os.makedirs(path)
argVector = {"input": "samples", "output": "outputs", "white noise": 0.5}

global audioInputFile, path


def parseArgs(argv):
    print("Parsing Arguments")

    i = 1
    while i < len(argv):
        try:
            if argv[i] == "-i":
                i = i + 1
                argVector["input"] = argv[i]
            elif argv[i] == "-o":
                i = i + 1
                argVector["output"] = argv[i]
            elif argv[i] == "-w":
                argVector["white noise"] = True
            else:
                print("Failed while parsing output. Please try again")
                exit()
        except:
            print("Failed while parsing output. Please try again")
            exit()
        i = i + 1

    print(argVector)

    global audioInputFile
    audioInputFile = glob(argVector["input"] + "/*.wav")

    global path
    path = os.path.join(".", argVector["output"])

    if not os.path.exists(path):
        os.makedirs(path)


def addWhiteNoise():
    audioData, sr = librosa.load(audioInputFile[0])
    max = np.max(audioData)
    min = np.min(audioData)

    outputData = audioData
    threshold = argVector["white noise"] * max
    negThreshold = threshold * min
    random.seed()

    for i in range(len(audioData)):
        rand = random.uniform(negThreshold, threshold)
        print(rand)
        outputData[i] = outputData[i] + rand

    outputPath = path + "/" + "white.wav"
    sf.write(outputPath, outputData, sr)


def main():
    parseArgs(sys.argv)
    addWhiteNoise()


if __name__ == "__main__":
    main()
