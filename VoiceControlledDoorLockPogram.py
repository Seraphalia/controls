import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import serial
import time
import pandas as pd
from scipy import interpolate


# -------------------------------------------------------------------------------------------

# Serial Connection with Arduino
arduino = serial.Serial('COM11', 9600)
time.sleep(5)

# -------------------------------------------------------------------------------------------
# Unlock Key Recording Section


# Start Recognizer
recognizer = sr.Recognizer()

# Calibrate for 3 seconds of ambient noise
with sr.Microphone() as source:
    print("Adjusting microphone for ambient noise...")
    recognizer.adjust_for_ambient_noise(source, duration=3)

# Start recording Unlock Audio
with sr.Microphone() as source:
    print("Please speak to record key...")
    unlock_data = recognizer.listen(source)
    print("Recording ended.")


# Store unlock audio as a Time Domain Signal
Unlock = np.frombuffer(unlock_data.frame_data, dtype=np.int16)

# Time array based on signal duration and sampling rate
unlock_time = np.linspace(0., len(Unlock) / unlock_data.sample_rate, num=len(Unlock))

# -------------------------------------------------------------------------------------------
# Unlock Key FFT Section


# Perform Fast Fourier Transform
unlock_freq_domain = np.fft.fft(Unlock)
# Get the magnitude of the Frequency Domain components
unlock_freq_domain = np.abs(unlock_freq_domain)

# Frequency array based on sampling rate and number of samples
sample_rate = unlock_data.sample_rate
unlock_freq = np.fft.fftfreq(len(Unlock), d=1/sample_rate)

# -------------------------------------------------------------------------------------------
# Unlock Key Bandpass Filter Section


# Bandpass filter to isolated wanted frequencies and attenuate noise
cutoff_low = 500  # Lower cutoff frequency
cutoff_high = 4000  # Upper cutoff frequency
nyquist = 0.5 * sample_rate
order = 5  # Filter order

# Design the bandpass filter
b, a = signal.butter(order, [cutoff_low/nyquist, cutoff_high/nyquist], btype='band')

# Apply bandpass filter to Frequency Domain Unlock Signal
filtered_unlock_freq_domain = signal.lfilter(b, a, unlock_freq_domain)

# -------------------------------------------------------------------------------------------
# Unlock Key Inverse FFT Section


# Perform Inverse Fourier Transform
filtered_unlock_time = np.fft.ifft(filtered_unlock_freq_domain)

# -------------------------------------------------------------------------------------------
# Unlock Key Visual Display Section


# Display the audio signals as a graph
plt.subplot(2,2,1)
plt.plot(unlock_time, Unlock)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain of [Unlock] Signal')

plt.subplot(2,2,2)
plt.plot(unlock_freq, unlock_freq_domain)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain of [Unlock] Signal')

plt.subplot(2,2,3)
plt.plot(unlock_time, filtered_unlock_time)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain of [Filtered Unlock] Signal')

plt.subplot(2,2,4)
plt.plot(unlock_freq, filtered_unlock_freq_domain)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain of [Filtered Unlock] Signal')


plt.tight_layout()
plt.show()
    
# -------------------------------------------------------------------------------------------
# Input Key Recording Section


user_input = input("Enter 1 to attempt unlocking. Enter any other key to end program: ")

if user_input == '1':
    
    # Calibrate for 3 seconds of ambient noise
    with sr.Microphone() as source:
        print("Adjusting microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=3)
    
    # Start recording Input Audio
    with sr.Microphone() as source:
        
        print("Please speak to record key...")
        input_data = recognizer.listen(source)
        print("Recording ended.")

    # Store input audio as a Time Domain Signal
    Input = np.frombuffer(input_data.frame_data, dtype=np.int16)

    # Time array based on signal duration and sampling rate
    input_time = np.linspace(0., len(Input) / input_data.sample_rate, num=len(Input))
    
    # -------------------------------------------------------------------------------------------
    # Input Key FFT Section
    
    
    # Perform Fast Fourier Transform
    input_freq_domain = np.fft.fft(Input)
    # Get the magnitude of the Frequency Domain components
    input_freq_domain = np.abs(input_freq_domain)

    # Frequency array based on sampling rate and number of samples
    sample_rate = input_data.sample_rate
    input_freq = np.fft.fftfreq(len(Input), d=1/sample_rate)
    
    # -------------------------------------------------------------------------------------------
    # Input Key Bandpass Filter Section


    # Apply the bandpass filter to Frequency Domain Input Signal
    filtered_input_freq_domain = signal.lfilter(b, a, input_freq_domain)
    
    # -------------------------------------------------------------------------------------------
    # Input Key Inverse FFT
    
    
    # Perform Inverse Fourier Transform
    filtered_input_time = np.fft.ifft(filtered_input_freq_domain)
    
    # -------------------------------------------------------------------------------------------
    # Input Key Visual Display Section
    
    
    # Display the [Filtered Unlock] and [Filtered Input] Audio Signals for visual comparison
    plt.subplot(2,2,1)
    plt.plot(unlock_time, filtered_unlock_time)
    plt.xlabel('Time (S)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain of [Filtered Unlock] Signal')

    plt.subplot(2,2,2)
    plt.plot(input_time, filtered_input_time)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain of [Filtered Input] Signal')

    plt.tight_layout()
    plt.show()
    
    # -------------------------------------------------------------------------------------------
    # Unlock-Input Correlation
    
    
    # Ensure that both filtered input and filtered unlock signals are in the same dimension
    if filtered_input_time.size < filtered_unlock_time.size:
        filtered_input_time = np.append(filtered_input_time, np.zeros(filtered_unlock_time.size - filtered_input_time.size))
    elif filtered_unlock_time.size < filtered_input_time.size:
        filtered_unlock_time = np.append(filtered_unlock_time, np.zeros(filtered_input_time.size - filtered_unlock_time.size))
    
    # Get the correlation coefficient
    signal_similarity = np.corrcoef(filtered_input_time, filtered_unlock_time)
    
    # Convert complex correlation coefficient into a real number
    if np.iscomplexobj(signal_similarity):
        correlation_matrix = np.abs(signal_similarity)
        
    # Extract correlation coefficient from correlation matrix
    similarity = correlation_matrix[0, 1]
    
    # Display correlation coefficient for user
    print(similarity)
    
    # Set threshold for correlation coefficient
    threshold = 0.1
        
    if similarity >= threshold:
        print("Unlocked")
        arduino.write(b'2')
    else:
        print("Locked")
        arduino.write(b'1')
    
    # -------------------------------------------------------------------------------------------
        
    
    max_length = max(len(Unlock), len(unlock_freq_domain), len(filtered_unlock_freq_domain), len(filtered_unlock_time))
        
    f = interpolate.interp1d(np.arange(len(Unlock)), Unlock)
    Unlock_interp = f(np.linspace(0, len(Unlock) - 1, max_length))
    
    f = interpolate.interp1d(np.arange(len(unlock_freq_domain)), unlock_freq_domain)
    unlock_freq_domain_interp = f(np.linspace(0, len(unlock_freq_domain) - 1, max_length))
    
    f = interpolate.interp1d(np.arange(len(filtered_unlock_freq_domain)), filtered_unlock_freq_domain)
    filtered_unlock_freq_domain_interp = f(np.linspace(0, len(filtered_unlock_freq_domain) - 1, max_length))
    
    f = interpolate.interp1d(np.arange(len(filtered_unlock_time)), filtered_unlock_time)
    filtered_unlock_time_interp = f(np.linspace(0, len(filtered_unlock_time) - 1, max_length))
    
    if np.iscomplexobj(filtered_unlock_time_interp):
        filtered_unlock_time_interp = np.abs(filtered_unlock_time_interp)
        
    df = pd.DataFrame({
        'Time Domain [Unlock] Signal': Unlock_interp,
        'Frequency Domain [Unlock] Signal': unlock_freq_domain_interp,
        'Frequency Domain [Filtered Unlock] Signal': filtered_unlock_freq_domain_interp,
        'Time Domain [Filtered Unlock] Signal': filtered_unlock_time_interp
    })
    
    df.to_excel('Data_Sheet.xlsx', index=False)
    
    # -------------------------------------------------------------------------------------------
    
else:
    print('Session ended.')
    exit()