
# Filtering signal with a butterworth low-pass filter and plotting the STFT of it with a Hann-Poisson window

Let's start by generating data. Let's assume that we have a sampling rate of 50 Hz over 10 seconds.


```python
import numpy as np

sample_rate = 50  # 50 Hz resolution
signal_lenght = 10*sample_rate  # 10 seconds
# Generate a random x(t) signal with waves and noise. 
t = np.linspace(0, 10, signal_lenght)
g = 30*( np.sin((t/10)**2) )
x  = 0.30*np.cos(2*np.pi*0.25*t - 0.2) 
x += 0.28*np.sin(2*np.pi*1.50*t + 1.0)
x += 0.10*np.sin(2*np.pi*5.85*g + 1.0)
x += 0.09*np.cos(2*np.pi*10.0*t)
x += 0.04*np.sin(2*np.pi*20.0*t)
x += 0.15*np.cos(2*np.pi*135.0*(t/5.0-1)**2)
x += 0.04*np.random.randn(len(t))
# Normalize between -0.5 to 0.5: 
x -= np.min(x)
x /= np.max(x)
x -= 0.5

```

## Filtering the generated data: 

Let's keep only what's below 15 Hz with a butterworth low-pass filter of order 4. Sharper cutoff can be obtained with higher orders. 
Butterworth low-pass filters has frequency responses that look like that according to their order: 

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Butterworth_Filter_Orders.svg/350px-Butterworth_Filter_Orders.svg.png" alt="Butterworth low-pass filter" style="width: 550px;" />

Let's proceed and filter the data. 


```python
from scipy import signal

import matplotlib.pyplot as plt
%matplotlib inline 


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# Filter signal x, result stored to y: 
cutoff_frequency = 15.0
y = butter_lowpass_filter(x, cutoff_frequency, sample_rate/2)

# Difference acts as a special high-pass from a reversed butterworth filter. 
diff = np.array(x)-np.array(y)

# Visualize
plt.figure(figsize=(11, 9))
plt.plot(x, color='red', label="Original signal, {} samples".format(signal_lenght))
plt.plot(y, color='blue', label="Filtered low-pass with cutoff frequency of {} Hz".format(cutoff_frequency))
plt.plot(diff, color='gray', label="What has been removed")
plt.legend()
plt.show()

```


![png](Filtering_files/Filtering_3_0.png)


## Plotting the spectrum with STFTs. 

Here, the [Hanning window](https://en.wikipedia.org/wiki/Window_function#Hann_.28Hanning.29_window) is used. 


```python
import scipy


def stft(x, fftsize=1024, overlap=4):
    # Short-time Fourier transform
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

# Here we don't use the ISTFT but having the function may be useful to some, 
# it is a bit modified than what's found on Stack Overflow: 
# def istft(X, overlap=4):
#     # Inverse Short-time Fourier transform
#     fftsize=(X.shape[1]-1)*2
#     hop = fftsize / overlap
#     w = scipy.hanning(fftsize+1)[:-1]
#     x = scipy.zeros(X.shape[0]*hop)
#     wsum = scipy.zeros(X.shape[0]*hop) 
#     for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
#         x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
#         wsum[i:i+fftsize] += w ** 2.
#     pos = wsum != 0
#     x[pos] /= wsum[pos]
#     return x

def plot_stft(x, interpolation='bicubic'):
    # Use 'none' interpolation for a sharp plot. 
    plt.figure(figsize=(11, 4))
    sss = stft(np.array(x), window_size, overlap)
    complex_norm_tape = np.absolute(sss).transpose()[::-1]
    plt.imshow(complex_norm_tape, aspect='auto', interpolation=interpolation, cmap=plt.cm.hot)
    # plt.yscale('log')
    plt.show()


window_size = 50  # a.k.a. fftsize
overlap = window_size  # This takes a maximal overlap


# Plots in the STFT time-frequency domain: 

print "x, original:"
plot_stft(x)

print "y, which is x filtered out:"
plot_stft(y)

print "Difference (notice changes in color due to plt's rescaling):"
plot_stft(diff)

```

    x, original:



![png](Filtering_files/Filtering_5_1.png)


    y, which is x filtered out:



![png](Filtering_files/Filtering_5_3.png)


    Difference (notice changes in color due to plt's rescaling):



![png](Filtering_files/Filtering_5_5.png)


## Other interesting stuff

If you want to know more about STFTs, FFTs and related signal processing, you may want to watch those videos I watched to understand the matter:

https://www.youtube.com/playlist?list=PLlp-GWNOd6m6gSz0wIcpvl4ixSlS-HEmr

## Connect with me

- https://ca.linkedin.com/in/chevalierg
- https://twitter.com/guillaume_che
- https://github.com/guillaume-chevalier/


```python
# Let's convert this notebook to a README as the GitHub project's title page:
!jupyter nbconvert --to markdown Filtering.ipynb
!mv Filtering.md README.md
```

    [NbConvertApp] Converting notebook Filtering.ipynb to markdown
    [NbConvertApp] Support files will be in Filtering_files/
    [NbConvertApp] Making directory Filtering_files
    [NbConvertApp] Making directory Filtering_files
    [NbConvertApp] Making directory Filtering_files
    [NbConvertApp] Making directory Filtering_files
    [NbConvertApp] Writing 5325 bytes to Filtering.md

