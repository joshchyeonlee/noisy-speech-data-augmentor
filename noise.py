import os
import sys
import random
import numpy as np
import soundfile as sf
from glob import glob

import librosa
import librosa.display

argVector = {"input": "samples", "output": "outputs", "white noise": 0.5}

global audioInputFile, outputPath


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

    global outputPath
    outputPath = os.path.join(".", argVector["output"])

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)


def addWhiteNoise():
    audioData, sr = librosa.load(audioInputFile[0])
    max = np.max(audioData)
    min = np.min(audioData)

    outputData = audioData
    threshold = argVector["white noise"] * max
    negThreshold = threshold * min

    for i in range(len(audioData)):
        rand = random.uniform(negThreshold, threshold)
        outputData[i] = outputData[i] + rand

    outputFilePath = (
        outputPath
        + "/"
        + "white_"
        + str(argVector["white noise"])
        + os.path.basename(audioInputFile[0])
    )
    sf.write(outputFilePath, outputData, sr)


def generateWhiteNoise(duration, sampleRate):
    sampleDuration = int(duration * sampleRate)
    return np.random.default_rng().uniform(-1, 1, sampleDuration)


def a1_coefficient(breakFreq, sampleRate):
    tan = np.tan(np.pi * breakFreq / sampleRate)
    return (tan - 1) / (tan + 1)


def allpassBasedFilter(input, cutoff, sampleRate, highpass=False, amplitude=1.0):
    allpassOutput = np.zeros_like(input)
    dn_1 = 0

    for i in range(input.shape[0]):
        a1 = a1_coefficient(cutoff[i], sampleRate)
        allpassOutput[i] = input[i] + dn_1
        dn_1 = input[i] - a1 * allpassOutput[i]

    if highpass:
        allpassOutput *= -1

    filterOutput = input + allpassOutput
    filterOutput *= 0.5
    filterOutput *= amplitude

    return filterOutput


# Based off of https://thewolfsound.com/allpass-based-lowpass-and-highpass-filters/
def createBrownNoise(sampleRate, duration):
    inputSignal = generateWhiteNoise(duration, sampleRate)
    cutoff = np.full(inputSignal.shape[0], 3000)
    output = allpassBasedFilter(inputSignal, cutoff, sampleRate, False, amplitude=0.1)
    sf.write("brown.wav", output, sampleRate)
    sf.write("white.wav", generateWhiteNoise(duration, sampleRate), sampleRate)


def secondOrderAllpassFilter(breakFreq, bandwidth, sampleRate):
    tan = np.tan(np.pi * bandwidth / sampleRate)
    c = (tan - 1) / (tan + 1)
    d = -np.cos(2 * np.pi * breakFreq / sampleRate)
    b = [-c, d * (1 - c), 1]
    a = [1, d * (1 - c), -c]

    return b, a


# modified from https://thewolfsound.com/allpass-based-bandstop-and-bandpass-filters/
def createBandPassFilter(sampleRate, duration, centerFrequency):
    sampleLength = int(sampleRate * duration)
    Q = 3

    inputSignal = generateWhiteNoise(duration, sampleRate)
    allpass = np.zeros_like(inputSignal)

    # previous and second last iteration inputs and outputs
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0

    for i in range(inputSignal.shape[0]):
        bandwidth = centerFrequency / Q
        b, a = secondOrderAllpassFilter(centerFrequency, bandwidth, sampleRate)
        x = inputSignal[i]

        y = b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2

        y2 = y1
        y1 = y
        x2 = x1
        x1 = x

        allpass[i] = y

    sign = -1
    output = 0.5 * (inputSignal + sign * allpass)
    sf.write("bandpass.wav", output, sampleRate)


def main():
    parseArgs(sys.argv)
    random.seed()
    # addWhiteNoise()
    createBrownNoise(44100, 5)
    createBandPassFilter(44100, 5, 700)


if __name__ == "__main__":
    main()
