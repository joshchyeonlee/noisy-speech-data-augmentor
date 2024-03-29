import os
import sys
import random
import numpy as np
import soundfile as sf
from glob import glob
from scipy.ndimage import shift

import librosa
import librosa.display
import argparse

global inputAudioFiles, outputPath, noiseAudioFiles, noisePath, noisePathExists

noisePathExists = True


def manageFiles(args):
    global inputAudioFiles
    
    if args.input:
        inputAudioFiles = glob(os.path.join(args.input, "*.wav"))
    else:
        inputAudioFiles = glob(os.path.join("samples/SpeechSamples", "*.wav"))
        
    if args.input is not None and not os.path.exists(args.input):
        print("Specified input directory does not exist. Exiting")
        sys.exit()

    global outputPath
    
    if args.output:
        outputPath = os.path.join(".", args.output)
    else:
        outputPath = os.path.join(".", "outputs")
        
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    global noiseAudioFiles, noisePath
    
    if args.noise:
        noisePath = os.path.join(".", args.noise)
        noiseAudioFiles = glob(os.path.join(noisePath, "*.wav"))
        if not os.path.exists(noisePath):
            global noisePathExists
            noisePathExists = False
            print("Failed parsing noise path. Omitting.")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--all', action='store_true', help="add mechanical whirr, next room, cutout, and delay effects. Combines all effects")
    parser.add_argument('-m','--mech', action='store_true', help="add mechanical whirr to mimic room fan noises")
    parser.add_argument('-r','--room', action='store_true', help="add muffled effect to mimic input audio coming from another room")
    parser.add_argument('-c','--cutout', action='store_true', help="add random cuts to the input audio to simulate packet drops")
    parser.add_argument('-d','--delay', action='store_true', help="add delay/echo effect to input audio to simulate feedback or room echo")
    parser.add_argument('-n','--noise', action='store', help="specify directory for ambient noise samples to be added to input")
    parser.add_argument('-i','--input', action='store', help="specify input directory for input audio speech files. Defaults to samples/SpeechSamples")
    parser.add_argument('-o','--output', action='store', help="specify output directory. Creates /output by default")
    
    args = parser.parse_args()
    
    manageFiles(args)
    
    return parser, args

def addWhiteNoise(audioData):
    max = np.max(audioData)
    min = np.min(audioData)

    outputData = audioData
    threshold = argVector["white noise"] * max
    negThreshold = threshold * min

    for i in range(len(audioData)):
        rand = random.uniform(negThreshold, threshold)
        outputData[i] = outputData[i] + rand


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
def lowPassFilter(inputSignal, sampleRate, cutoffFrequency, amplitude):
    cutoff = np.full(inputSignal.shape[0], cutoffFrequency)
    output = allpassBasedFilter(
        inputSignal, cutoff, sampleRate, False, amplitude=amplitude
    )
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
def cutoutEffect(audioData, probability=0.0003):
    length = len(audioData)
    cut = np.random.choice([0, 1], length, p=[probability, 1 - probability])
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


def delayFilter(audioData, sampleRate, delayTime=500, feedback=0.4):
    geometric = np.geomspace(1, 2, audioData.shape[0])
    geometric -= 1

    delayPoint = int((delayTime * sampleRate) / 1000)
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


def phoneEffect(audioData, sr):
    max = np.max(audioData)
    audioData = cutoutEffect(audioData)
    audioData = bandPassFilter(audioData, sr, 2000, 3)
    audioData = bandPassFilter(audioData, sr, 400, 3)

    newMax = np.max(audioData)
    audioData *= max / newMax
    return audioData


def generateMechanicalWhirr(frequency, duration, sampleRate, lowpassFrequency):
    audioData = generateSineWave(frequency, duration, sampleRate)
    audioData += generateSineWave(int(frequency / 2), duration, sampleRate)
    audioData = audioData + generateWhiteNoise(duration, sampleRate)
    audioData = lowPassFilter(audioData, sampleRate, lowpassFrequency, 0.001)
    audioData = audioData * 0.1
    return audioData


def generateSineWave(fundamentalFrequency, duration, sampleRate):
    samples = np.linspace(0, duration, int(duration * sampleRate), endpoint=False)

    signal = np.sin(2 * np.pi * fundamentalFrequency * samples)
    signal *= 32767
    signal = np.int16(signal)

    return signal


def nextRoomEffect(audioData, sampleRate):
    cutoffFrequency = 300
    audioData = lowPassFilter(audioData, sampleRate, cutoffFrequency, 0.1)
    return audioData


def addBackgroundNoise(inputAudioData, backgroundNoise, balance):
    adjusted = adjustLength(inputAudioData, backgroundNoise)
    adjustedBackgroundNoise = adjusted * balance
    outputData = inputAudioData + adjustedBackgroundNoise
    return outputData


def adjustLength(audio1, audio2):
    len1 = audio1.shape[0]
    len2 = audio2.shape[0]
    diff = len1 - len2

    if diff <= 0:
        return audio2

    output = np.pad(audio2, (0, diff), "constant")
    return output


def createOutputFile(outputPath, noiseType, outputFileName):
    outputFilePath = os.path.join(outputPath, noiseType)

    if not os.path.exists(outputFilePath):
        os.makedirs(outputFilePath)

    outputFilePath = os.path.join(outputFilePath, outputFileName)

    return outputFilePath


def main():
    parser, args = parseArgs()
        
    if not (args.all or args.mech or args.room or args.cutout or args.delay):
        print("\nOne of -a, -m, -r, -c, or -d required to run program. See command details below\n")
        parser.print_help()
        sys.exit()
    
    random.seed()
    
    for audioFile in inputAudioFiles:
        audioData, sr = librosa.load(audioFile)
        audioData = normalize(audioData)
        mixedData = audioData
        fileName = audioFile.split("/")[-1]
        fileNameParsed = fileName.split(".")[0]

        if (args.noise or args.all) and noisePathExists:
            for noiseAudio in noiseAudioFiles:
                noiseAudioData, noiseSR = librosa.load(noiseAudio)
                randomBalance = round(random.uniform(0.1, 1.0), 2)

                if args.all:
                    mixedData = addBackgroundNoise(mixedData, noiseAudioData, randomBalance)
                
                if args.noise:
                    outputAudioData = addBackgroundNoise(audioData, noiseAudioData, randomBalance)
                
                    sampleRate = sr if sr < noiseSR else noiseSR
                    noiseAudioFileName = noiseAudio.split("/")[-1].split(".")[0]
                    outputFileName = (
                        fileNameParsed
                        + "+"
                        + noiseAudioFileName
                        + "_bal="
                        + str(randomBalance)
                        + ".wav"
                    )

                    outputFilePath = createOutputFile(outputPath, "noise", outputFileName)

                    sf.write(outputFilePath, outputAudioData, sampleRate)

        if args.mech or args.all:
            
            randomFreq = random.randint(40, 100)
            audioLen = audioData.shape[0] / sr
            randLowPassFreq = random.randint(100, 200)
            whirrAudio = generateMechanicalWhirr(
                randomFreq, audioLen, sr, randLowPassFreq
            )
            randomBalance = round(random.uniform(0.1, 0.3), 3)

            if args.all:
                mixedData = addBackgroundNoise(mixedData, whirrAudio, randomBalance)

            if args.mech:
                outputFileName = (
                    fileNameParsed
                    + "+mechanicalWhirr_"
                    + str(randomFreq)
                    + "Hz+lowPass_"
                    + str(randLowPassFreq)
                    + "Hz.wav"
                )
                
                outputAudioData = addBackgroundNoise(audioData, whirrAudio, randomBalance)
                outputFilePath = createOutputFile(outputPath, "mech_whirr", outputFileName)

                sf.write(outputFilePath, outputAudioData, sr)

        if args.room or args.all:
            if args.all:
                mixedData = nextRoomEffect(mixedData, sr)
            
            if args.room:
                roomAudio = nextRoomEffect(audioData, sr)
            
                outputFileName = fileNameParsed + "+nextRoom.wav"
                outputFilePath = createOutputFile(outputPath, "room", outputFileName)
                sf.write(outputFilePath, roomAudio, sr)

        if args.cutout or args.all:
            cutoutProbability = round(random.uniform(0.0002, 0.0005), 5)
            
            if args.all:
                mixedData = cutoutEffect(mixedData, cutoutProbability)
            
            if args.room:
                cutoutAudio = cutoutEffect(audioData, cutoutProbability)
                outputFileName = (
                    fileNameParsed + "+cutout_" + str(cutoutProbability) + ".wav"
                )
                outputFilePath = createOutputFile(outputPath, "cutout", outputFileName)

                sf.write(outputFilePath, cutoutAudio, sr)

        if args.delay or args.all:
            delayTime = random.randint(300, 700)
            feedback = round(random.uniform(0.1, 0.7), 2)
            
            if args.all:
                mixedData = delayFilter(mixedData, sr, delayTime, feedback)

            if args.delay:
                delayAudio = delayFilter(audioData, sr, delayTime, feedback)
                outputFileName = (
                    fileNameParsed
                    + "+delay_"
                    + str(delayTime)
                    + "+"
                    + str(feedback)
                    + ".wav"
                )
                outputFilePath = createOutputFile(outputPath, "delay", outputFileName)

                sf.write(outputFilePath, delayAudio, sr)

        if args.all:
            outputFilePath = createOutputFile(outputPath, "mixed", fileNameParsed + "mixed.wav")
            sf.write(outputFilePath, mixedData, sr)


if __name__ == "__main__":
    main()
