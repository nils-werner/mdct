#! /usr/bin/env python2.7
'''
mdct_test.py

 Uses MDCT and STFT libraries by Nils Werner, e.g.
 		https://github.com/nils-werner/mdct
 		http://stft.readthedocs.io/en/latest/
 	...both installed using pip, e.g. "pip install mdct"

 	python2.7 is needed for scikits.audiolab.  Otherwise this is python3 code
'''

import numpy as np 
import matplotlib.pyplot as plt
import mdct, stft
import scipy.fftpack
from scikits.audiolab import Sndfile
import os.path

# unlike scipy.io.wavfile, the following also works for 24-bit audio files
def read_audio_file(file_path):
    if os.path.isfile(file_path):
        f = Sndfile(unicode(file_path), 'r')
        wav_data = np.array(f.read_frames(f.nframes), dtype=np.float32)
        samplerate = f.samplerate
        f.close()

        # just for simplicity right now: take left channel of stereo, so we only have mono
        if (len(wav_data.shape) > 1):    
            wav_data = wav_data[:,0]
 
        return samplerate, wav_data


# Load the audio file
rate, signal = read_audio_file("test_audio.wav")  
tmax = signal.shape[0]/rate
time = np.linspace(0,tmax, signal.shape[0])
print("sample rate = ",rate,", tmax = ",tmax)

fig=plt.figure()

# Plot time series 
ax1 = plt.subplot(411)
ax1.set_title("Time Series")
ax1.set_xlim(0,tmax)
plt.plot(time,signal) 
print("signal.shape = ",signal.shape)

# Plot 'default' matplotlib spectrogram, using overlap comparable to mdct lapping
ax2 = plt.subplot(412)  
ax2.set_title("Standard Matplotlib Spectrogram")
ax2.set_xlim([0,tmax])
framelength = 1024
spectrum, freqs, ts, im = ax2.specgram(signal, NFFT=framelength,Fs=rate,noverlap=framelength/2,cmap=plt.cm.gist_heat)
#cbar = fig.colorbar(im)
print("spectrum: shape = ",spectrum.shape,", dtype = ",spectrum.dtype)

# Plot stft library version
ax3 = plt.subplot(413) 
eps = 1e-10
ax3.set_title("STFT lib Spectrogram")
stft_trans = stft.spectrogram(signal, framelength=framelength)
logmag = np.flipud(np.log(eps+stft_trans.real**2 + stft_trans.imag**2)) # flip up/down to put high freqs on top
plt.imshow(logmag, extent=[0,tmax,0,512], aspect='auto',cmap=plt.cm.gist_heat)
print("stft_trans: shape = ",stft_trans.shape,", dtype = ",stft_trans.dtype)

# Plot mdct version
ax4 = plt.subplot(414) 
ax4.set_title("MDCT lib Spectrogram")
mdct_trans = mdct.mdct(signal, framelength=framelength)
logmag = np.flipud(np.log(eps+mdct_trans**2))				# flip up/down to put high freqs on top
plt.imshow(logmag, extent=[0,tmax,0,512], aspect='auto',cmap=plt.cm.gist_heat)
print("mdct_trans: shape = ",mdct_trans.shape,", dtype = ",mdct_trans.dtype)

plt.show()


