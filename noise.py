import os
import sys
import random
import numpy as np
import soundfile as sf
from glob import glob
from scipy.ndimage import shift

import librosa
import librosa.display

argVector = {"input": "samples", "output": "outputs", "white noise": 0.5, "band pass": 400}

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
                i = i + 1
                argVector["white noise"] = argv[i]
            elif argv[i] == "-b":
                i = i + 1
                argVector["band pass"] = argv[i]
            else:
                print("Failed while parsing input. Please try again")
                exit()
        except:
            print("Failed while parsing input. Please try again")
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
        allpassOutput[i] = a1 * input[i] + dn_1
        dn_1 = input[i] - a1 * allpassOutput[i]

    if highpass:
        allpassOutput *= -1

    filterOutput = input + allpassOutput
    filterOutput = filterOutput * 0.5
    filterOutput = filterOutput * amplitude

    return filterOutput


# Based off of https://thewolfsound.com/allpass-based-lowpass-and-highpass-filters/
def lowPassFilter(inputSignal, sampleRate, cutoffFrequency):
    # inputSignal = generateWhiteNoise(duration, sampleRate)
    cutoff = np.full(inputSignal.shape[0], cutoffFrequency)
    output = allpassBasedFilter(inputSignal, cutoff, sampleRate, False, amplitude=0.1)
    return output

def secondOrderAllpassFilter(breakFreq, bandwidth, sampleRate):
    tan = np.tan(np.pi * bandwidth / sampleRate)
    c = (tan - 1) / (tan + 1)
    d = -np.cos(2 * np.pi * breakFreq / sampleRate)
    b = [-c, d * (1 - c), 1]
    a = [1, d * (1 - c), -c]

    return b, a


# modified from https://thewolfsound.com/allpass-based-bandstop-and-bandpass-filters/
def bandPassFilter(inputSignal, sampleRate, centerFrequency, Q):
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
    return output


def normalize(audioData):
    max = np.max(audioData)
    min = np.min(audioData)
    
    peak = np.absolute(max) if np.absolute(max) > np.absolute(min) else np.absolute(min)
    
    scale = 1 / peak
    audioData *= scale
    return audioData
    
# p < 0.001 recommended
def cutoutEffect(audioData, probability = 0.0003):
    length = len(audioData)
    cut = np.random.choice([0,1],length,p=[probability,1 - probability])
    i = 0
    
    while i < length:
        if cut[i] == 0:
            randomLength = random.randrange(int(length * probability * 50))
            for x in range(randomLength):
                if i >= length:
                    break
                cut[i] = 0
                i += 1
        i += 1
    
    audioData *= cut
    return audioData
    
    
def delayFilter(delayTime = 500, feedback = 0.4):
    audioData, sr = librosa.load(audioInputFile[0])

    geometric = np.geomspace(1, 2, audioData.shape[0])
    geometric -= 1

    delayPoint = int((delayTime * sr) / 1000)
    iteration = 1
    
    rawAudio = np.copy(audioData)
    
    while feedback < 1:
        delayAmount = geometric[int(len(geometric) * feedback)]
        delayAudio = rawAudio * delayAmount
        delayAudio = shift(delayAudio, (delayPoint * iteration), cval=0)
        
        audioData += delayAudio
        feedback *= feedback + 1
        iteration += 1
    
    return audioData
    
def phoneEffect():
    audioData, sr = librosa.load(audioInputFile[0])
    max = np.max(audioData)
    print(max)
    audioData = cutoutEffect(audioData)
    audioData = bandPassFilter(audioData, sr, 2000, 3)
    audioData = bandPassFilter(audioData, sr, 400, 3)
    
    newMax = np.max(audioData)
    audioData *= (max / newMax)
    sf.write("phone.wav", audioData, sr)

def generateMechanicalWhirr(frequency, duration, sampleRate, lowpassFrequency):
    audioData = generateSineWave(frequency, duration, sampleRate)
    audioData += generateSineWave(int(frequency * 1.5), duration, sampleRate)
    audioData = audioData + generateWhiteNoise(duration, sampleRate)
    audioData = lowPassFilter(audioData, sampleRate, lowpassFrequency)
    audioData = audioData + generateWhiteNoise(duration, sampleRate)
    # pitchShifted = librosa.effects.pitch_shift(audioData, sr=sampleRate, n_steps=-1, bins_per_octave=12)
    sf.write("whirr.wav", audioData, sampleRate)
    # sf.write("pitchedWhirr.wav", pitchShifted, sampleRate)

def generateSineWave(fundamentalFrequency, duration, sampleRate):
    samples = np.linspace(0, duration, int(duration * sampleRate), endpoint=False)
    
    signal = np.sin(2 * np.pi * fundamentalFrequency * samples)
    signal *= 32767
    signal = np.int16(signal)
    
    return signal

def main():
    parseArgs(sys.argv)
    random.seed()
    # delayFilter(500, 0.4)
    # phoneEffect()
    # cutoutEffect()
    # normalize()
    # addWhiteNoise()
    # lowpassFilter(44100, 5)
    # bandPassFilter(44100, 5, 700, 3)
    generateMechanicalWhirr(60, 5, 44100, 300)


if __name__ == "__main__":
    main()
