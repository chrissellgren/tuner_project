#!/usr/bin/env python3

# Final project for Phys 129L at UCSB, Winter 2023
# This tuner takes in user-specified seconds of microphone input and prints feedback about the intonation of the note.

######################################################

#### relevant packages ###

import numpy as np
import scipy.interpolate
import scipy.signal
import pyaudio 
import time
import tkinter as tk

######################################################

### Launch Graphical Interface ####

# create interface
root = tk.Tk()
root.title("Guitar Tuner") # title

# create function to take user input 
def save_input():
    # attempt to save the input as a float
    try:
        samplingtime = float(entry.get()) # seconds
        print("Input saved:", samplingtime)
    # if the input cannot be converted, print error to console
    except ValueError:
        print("Invalid input")

# length of sampling window, in seconds
samplingtime = tk.StringVar()
samplingtime = 4  #in case no sampling time is entered

# create the input box
entry = tk.Entry(root)
entry.pack()

# create a button to save the input
button1 = tk.Button(root, text="Save Input", command=save_input)
button1.pack()

######################################################

###  Set initial parameters for global variables that will be called in multiple functions ###

# chunk size in sampling
samplesize = 2048 

# sampling rate, how many samples will be taken per second, determined from PyAudio forum
samplespersec = 48000 

totalsamples = samplespersec * samplingtime

# convert frequency domain to Hz (will be used later)
freqconversion = samplespersec/samplesize

######################################################

### Create Function for Recording and Analyzing Audio ###

def take_audio():
    
    # start recording data
    print("Detecting Audio...")
    t0 = time.perf_counter()

    # open the audio data from microphone
    # this syntax I retrieved from the PyAudio documentation: https://people.csail.mit.edu/hubert/pyaudio/docs/#class-pyaudio-stream
    p = pyaudio.PyAudio()
    audiostream = p.open(format = pyaudio.paInt16, # 16 bit integer is data type
                                 channels = 1, 
                                 rate = samplespersec, 
                                 frames_per_buffer = samplesize, 
                                 input = True)

    # wait for microphone to collect enough data
    time.sleep(samplingtime)

    # print how long the sampling occurred for
    t = time.perf_counter() - t0
    print("Sampled %.3f seconds of microphone data" % t)

    # extract data from audio stream
    numdatapoints = totalsamples//samplesize # must be an int
    data = audiostream.read(samplesize,exception_on_overflow=False)[:int(totalsamples)]

    # close stream
    audiostream.close()

    # the data is saved as a byte string, so convert to 16-bit integer
    # frombuffer used because fromstring does not handle unicode inputs
    data = np.frombuffer(data,dtype=np.int16)

    # convert the 16 bit integers into floats
    # also normalize the floats to range from 0.0 to 1.0
    data = data / (2**15)

    #############################################################

    ### Analyze Data ###

    # now  the audio data is in floats
    # we can use a fourier transform to analyze the frequency spectrum of the data
    freqspec = np.fft.fft(data)

    # in signal processing it is a good idea to smooth out the ends of the data set
    # this protects against discontinuities
    # the hamming window is the most common choice in signal processing
    window = scipy.signal.hamming(samplesize)

    # take absolute value to get intensity as a function of frequency
    smoothedspec = np.abs(freqspec * window)

    # limit frequencies to detectable range
    lowestfreq = 80 # Hz, around threshold of lowest note on a guitar
    highestfreq = 1200 # Hz, around highest note a guitar could play
    # create an list of indices for the smoothedspec that fall in this range
    goodindices = [i for i in range(len(smoothedspec)) if i*freqconversion > lowestfreq and i*freqconversion < highestfreq]
    # create first and last indices
    i1, i2 = goodindices[0], goodindices[-1] 

    # now we isolate the frequency with the peak intensity, searching in the acceptable range
    # add i1 so that bestindex corresponds to an index in entire smoothedspec array
    bestindex = smoothedspec[i1:i2].argmax() + i1

    # note that the true frequency we're looking for is likely not exactly at this point
    # use an interpolation around the peak intensity to determine the best freq
    # start by collecting 3 intensities near peak
    xvalues = [i for i in range(bestindex -1, bestindex + 2,1)]
    intensities_nearpeak = [smoothedspec[i] for i in xvalues]
    # tuner must respond to log of intensities -> sound detection of human ear
    yvalues = np.log(intensities_nearpeak)

    # create an interpolation function
    interpolation = scipy.interpolate.interp1d(xvalues,yvalues,kind='quadratic')
    # find the peak of the interpolation in this this range
    xrange = np.linspace(xvalues[0],xvalues[2],10000)
    yrange = interpolation(xrange)
    peakindex = xrange[yrange.argmax()]

    # now we convert the index of best frequency to actual frequency
    bestfreq = peakindex * freqconversion # Hz
    
    rawfreqstatement = "The peak frequency detected was %.1f Hz" % bestfreq
    # print out the text for the raw frequency in a TkInter label
    rawfreq_label.config(text=rawfreqstatement)

    print(rawfreqstatement)
    
    # we have the frequency of the pitch we're looking for
    # however, it is possible that this frequency (greatest intensity) is actually a harmonic of the one we're looking for
    # to protect against this possibility, we just use the note, which can correspond to any harmonic of the pitch

    # to find the note, we convert the frequency to cents, 1/100 of a half-step
    # cents conversion (found here: http://hyperphysics.phy-astr.gsu.edu/hbase/Music/cents.html)
    A_4 = 440 # Hz, Western tuning convention that pitch measured against
    distancefromA4_cents = 1200 * np.log2(A_4/bestfreq)/100
    # now we have how far from A_4 our pitch is in steps
    # to find which note it is, take modulo of distance from octave
    cents = distancefromA4_cents % 12
    
    return cents, smoothedspec, data


#################################################################

### Create Function to Create Graphics ###

def produce_graphics(smoothedspec, data):
    # make plots of the data 
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #necessary for TkInter GUI

    # raw audio data
    x = np.linspace(0,samplingtime,samplesize)
    f1 , (ax1, ax2) = plt.subplots(2)
    ax1.plot(x,data)
    ax1.set_xlabel("Recording Time (s)")
    ax1.set_ylabel("Normalized Sound Amplitude")
    ax1.set_title("Raw Audio Data")
    
      # frequency spectrum
    freqs = np.arange(0,(len(smoothedspec)))*freqconversion
    ax2.plot(freqs,smoothedspec)
    ax2.set_xlim(0,1000) #Hz
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Relative Intensity")
    ax2.set_title("Frequency Spectrum")
    f1.subplots_adjust(hspace=1)

    # plot in GUI

    # now create TkInter canvas widget to display the plot
    canvas1 = FigureCanvasTkAgg(f1, master=root)
    canvas1.draw()
    canvas1.get_tk_widget().pack()
    canvas_list.append(canvas1)


#################################################################

### Determine Note ####

# compare the best frequency in cents with the values for each note on scale
A1 = 0 #tuning frequency
A2 = 12
# Asharp is11 half steps away, modulo of cent value would be 11
B = 10 #10 half steps away...
# C = 9, but only use values corresponding to strings on a guitar
#Csharp = 8
D = 7
#Dsharp = 6
E = 5
#F = 4
#Fsharp = 3
G = 2
#Gsharp = 1


# create function to check every single possible note

# shortcuts for printing whether note was sharp or flat
def sharpstatus():
    return "you are sharp by more than 20 cents!"

def flatstatus():
    return "you are flat by more than 20 cents!"

def intune():
    return "you are in tune to within 20 cents!"

def checknote(cents):
    """This function takes in an input frequency in cents
    and returns a string declaring what note it is and how in tune
    the note was."""
    
    # check A
    if abs(cents - A1) < 1.2: 
        note = "A"
        if np.abs(cents - A1) < 0.2:
            status = intune()
        elif (cents - A1) < 0:
            status = sharpstatus()
        elif (cents - A1) > 0:
            status = flatstatus()
    # need to check A twice, in case the cents value is between 11 and 12
    if abs(cents - A2) < 1.2: 
        note = "A"
        if np.abs(cents - A2) < 0.2:
            status = intune()
        elif (cents - A2) < 0:
            status = sharpstatus()
        elif (cents - A2) > 0:
            status = flatstatus()

    # B 
    elif abs(cents - B) < 1.2:
        note = "B"
        if np.abs(cents - B) < 0.2:
            status = intune()
        elif (cents - B) < 0:
            status = sharpstatus()
        elif (cents - B) > 0:
            status = flatstatus()

    # D
    elif abs(cents - D) < 1.2:
        note = "D"
        if np.abs(cents - D) < 0.2:
            status = intune()
        elif (cents - D) < 0:
            status = sharpstatus()
        elif (cents - D) > 0:
            status = flatstatus()

    # E
    elif abs(cents - E) < 1.2:
        note = "E"
        if np.abs(cents - E) < 0.2:
            status = intune()
        elif (cents - E) < 0:
            status = sharpstatus()
        elif (cents - E) > 0:
            status = flatstatus()

    # G
    elif abs(cents - G) < 1.2:
        note = "G"
        if np.abs(cents - G) < 0.2:
            status = intune()
        elif (cents - G) < 0:
            status = sharpstatus()
        elif (cents - G) > 0:
            status = flatstatus()

    else:
        note = "an unidentified note"
        status = "something went wrong! Please try again."


    outcome = "You played " + note + " and " + status
    result_label.config(text = outcome)
    return outcome

#################################################################

### Define function to do all steps ###
def tuner():
    cents, spec, data = take_audio()
    print(checknote(cents))
    produce_graphics(spec, data)


### Graphical interface for the button and results ###

# create list of canvases that our plot will be added to
# this will be used to clear out plots when resampling
canvas_list = []

# create button to launch program
button = tk.Button(root, text="Begin Sampling", command=tuner)
button.pack()

# label for raw frequency detection
rawfreq_label = tk.Label(root, text='')
rawfreq_label.pack()

# label for note and intonation
result_label = tk.Label(root, text='')
result_label.pack()


# fxn to clear widgets
def clear_plots():
    # reference empty list made earlier, appended to in graphics function
    for canvas in canvas_list:
        canvas.get_tk_widget().pack_forget()
    # Clear the canvas list
    canvas_list.clear()

#button to clear canvases
clear_button = tk.Button(root, text="Clear Plots", command=clear_plots)
clear_button.pack()

root.mainloop()