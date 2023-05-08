Hospital ambient noise might be useful for performing many healthcare simulations. It is recorded from different places (corridor, waiting room etc.) of a hospital. This dataset contains 2 folders. 

1) Hospital noise original: It contains 562 chunks of audio, each of 5s. The data is sampled at 44.1 KHz.

2) Hospital noise filtered resampled: It also contains 562 chunks of audio, each of 5s. Since FFT shows the major frequency components to be within 500 Hz, the data is filtered with  a 3rd order Butterworth low pass filter having cut off at 500 Hz. Then, it is resampled to 1000 Hz, following the Nyquist criteria. This processing has been done so that it can easily used without costing much memory while maintaining signal fidelity.  