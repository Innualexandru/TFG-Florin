# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:38:12 2019

@author: Innu
"""
from scipy import signal, interpolate
from scipy.signal import butter, lfilter
import math as m
import numpy as np #librería matemática de py
import pandas as pd #librería lectura de datos o frames (.npy y .csv) 
import matplotlib.pyplot as plt
import seaborn as sns


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_high_pass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_high_pass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

order = 5
cutoff_l = 35
cutoff_h = 0.2
fs = 500


# Abro el fichero donde tengo guardado os segmentos
with open('info_segments_2.csv') as file:
    datos = pd.read_csv(file, delimiter= ',', header=0) #leo el fichero y vuelco en datos el valor
    
    
data_signals = np.load('segments_2.npy') #cargo las señales en data 


#------------ESTADÍSTICAS TEMPORALES (C)-----------------------#

# Voy a trabajar primero con las 2500 muestras de un sujeto


senial_indep_2500_muestras = data_signals[1,2,:]
senial_indep_2500_muestras_dislexia = data_signals[22,1500,:]


# Plot de la señal sin filtrar:
sns.set()
sf = 500
time_dislex = np.arange(senial_indep_2500_muestras_dislexia.size)/sf
time = np.arange(senial_indep_2500_muestras.size)/sf



# Dibujo la señal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time_dislex, senial_indep_2500_muestras_dislexia, lw=1.5, color='red')
plt.plot(time, senial_indep_2500_muestras, lw=1.5, color='k')
plt.xlabel('Time[sec]')
plt.ylabel('Amplitud')
plt.xlim([time_dislex.min(), time_dislex.max()])
plt.title('Señal sin filtrar')
sns.despine()

# Filtrado de la señal

y_filtrada = butter_lowpass_filter(senial_indep_2500_muestras, cutoff_l, fs, order)
y_filtrada = butter_highpass_filter(y_filtrada, cutoff_h, fs, order)

y_filtrada_dislex = butter_lowpass_filter(senial_indep_2500_muestras_dislexia, cutoff_l, fs, order)
y_filtrada_dislex = butter_highpass_filter(y_filtrada_dislex, cutoff_h, fs, order)

senial_indep_2500_muestras = y_filtrada
senial_indep_2500_muestras_dislexia = y_filtrada_dislex


# from PyEMD import EMD,visualisation
from PyEMD.EMD import EMD
from PyEMD.visualisation import Visualisation


# Perform decomposition
print("Performing decomposition... ")

emd = EMD()
emd.emd(senial_indep_2500_muestras, max_imf=5)

emd_dis = EMD()
emd_dis.emd(senial_indep_2500_muestras_dislexia, max_imf=5)


imfs,res = emd.get_imfs_and_residue()
imfs_dis,res_dis = emd_dis.get_imfs_and_residue()


vis = Visualisation()

print("Sujeto Normal")
vis.plot_imfs(imfs=imfs, residue=res, t=time, include_residue=False)
vis.plot_instant_freq(time, imfs=imfs)
vis.show()

print("Sujeto Dislexia")
vis.plot_imfs(imfs=imfs_dis, residue=res_dis, t=time_dislex, include_residue=False)
vis.plot_instant_freq(time_dislex, imfs=imfs_dis)
vis.show()


# Una vez extraída la señal hago la media: 

ut = senial_indep_2500_muestras.mean() #media

#----dislexia----#

ut_dislex = senial_indep_2500_muestras_dislexia.mean()

#Desviación típica: 

#Preparo el momento central de orden 2 

momento_central = senial_indep_2500_muestras-ut

momento_central_orden_dos = pow(momento_central,2)

media_ot = momento_central_orden_dos.mean()

ot = m.sqrt(media_ot) #Desviación típica

#-----dislexia----#


momento_central_dislex = senial_indep_2500_muestras_dislexia-ut_dislex

momento_central_orden_dos_dislex = pow(momento_central_dislex,2)

media_ot_dislex = momento_central_orden_dos_dislex.mean()

ot_dislex = m.sqrt(media_ot_dislex)




# Skewness

aux_1 = momento_central/ot 

Bt = pow(aux_1,3).mean()

#----dislexia----#

aux_1_dislex = momento_central_dislex/ot_dislex 

Bt_dislex = pow(aux_1_dislex,3).mean()


""" Hasta aquí el análisis en tiempo, a continuación pinto la gráfica de la señal """


sns.set(font_scale=1.2)
# Define sampling frequency and time vector (Reduce la escala del muestreo)
sf = 500. #Según valga sf puedo hacer más pequeña o no mi ventana
# Mi ventana es de 500 debido a que tengo en 5 segundos 2500 muestras

""" Fs = N/t así que como N = 2500 y t = 5s sale una freq de muestreo de 500 Hz -> es sf"""


#Señal filtrada a continuación:

time_dislex_filtrada = np.arange(y_filtrada_dislex.size)/sf
time_filtrada = np.arange(y_filtrada.size)/sf
# Dibujo la señal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time_dislex_filtrada, y_filtrada_dislex, lw=1.5, color='red')
plt.plot(time_filtrada, y_filtrada, lw=1.5, color='k')
plt.xlabel('Time[sec]')
plt.ylabel('Amplitud')
plt.xlim([time_dislex_filtrada.min(), time_dislex_filtrada.max()])
plt.title('Señal filtrada')
sns.despine()


#-----------ANÁLISIS ESPECTRAL (D)-------------------------#

"""Debido a que en un amplio rango de señal es muy dificil que la señal sea una suma
periódica sinusoidal se emplea el método de Weltch ya que trabaja con pequeños trozos de señal
(ventanas) para calcular el espectro"""

win = 1.25*sf #el tamaño de la ventana es cuantas veces quiero leer la muestra
freqs_dislex, psd_dislex = signal.welch(senial_indep_2500_muestras_dislexia, sf, nperseg=win)
freqs, psd = signal.welch(senial_indep_2500_muestras, sf, nperseg=win)

"""The freqs vector contains the x-axis (frequency bins) and the psd vector contains the y-axis (power spectral density). 
The units of the power spectral density, when working with EEG data, is usually micro-Volts-squared per Hz """


# Plot the power spectrum (densidad) -> P(w)

sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(freqs, psd, color='k', lw=2)
plt.plot(freqs_dislex, psd_dislex, color='r', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd_dislex.max() * 1.1])
plt.title("Welch's periodogram. Sujeto de Control (negro) vs Sujeto Dislexia (rojo)")
plt.xlim([0, 100])
sns.despine()




# Centroide espectral:
w = freqs*2*m.pi
aux_2 = psd*w

Cs = sum(aux_2)/sum(psd) #centroide espectral


#----dislexia---#

w_dislex = freqs_dislex*2*m.pi
aux_2_dislex = psd_dislex*w_dislex

Cs_dislex = sum(aux_2_dislex)/sum(psd_dislex)


# Coeficiente de variacion espectral

aux_3 = w-Cs
aux_4 = pow(aux_3,2)
aux_5 = psd*aux_4
os_cuadrado = sum(aux_5)/sum(psd)

os = m.sqrt(os_cuadrado)

#---dislexia---#

aux_3_dislex = w_dislex-Cs_dislex
aux_4_dislex = pow(aux_3_dislex,2)
aux_5_dislex = psd_dislex*aux_4_dislex
os_cuadrado_dislex = sum(aux_5_dislex)/sum(psd_dislex)

os_dislex = m.sqrt(os_cuadrado_dislex)

# Spectral Skew

aux_6 = pow(aux_3/os,3)*psd

Bs = sum(aux_6)/sum(psd)


#----dislexia---#

aux_6_dislex = pow(aux_3_dislex/os_dislex,3)*psd_dislex

Bs_dislex = sum(aux_6_dislex)/sum(psd_dislex)


# Por último concatenamos estos resultados en un único vector F para su estudio:
 
F = [ut, ot, Bt, Cs, os, Bs]
F_dislex = [ut_dislex, ot_dislex, Bt_dislex, Cs_dislex, os_dislex, Bs_dislex]


sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(F, color='k', lw=2)
plt.plot(F_dislex, color='r', lw=2)
plt.plot(F, 'ro')
plt.plot(F_dislex, 'bo')
plt.xlabel('Caracteríticas')
plt.ylabel('Valor de las caracterísitcas')
plt.title("Sujeto de Control (negro) vs Sujeto Dislexia (rojo)")
plt.xlim([-0.5, 6])
sns.despine()




fs = 500
N = 2500
x = senial_indep_2500_muestras
y = senial_indep_2500_muestras_dislexia

from pylab import savefig, specgram
import sys, string 
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import matplotlib.mlab as mlab
from os import makedirs, path


# Calculate spectrogram 
secsPerFFT = 1 # Hz
overlap = 0.9 # fractional overlap
Fs = 500
NFFT = int(round(Fs*secsPerFFT))
noverlap = int(round(overlap*NFFT))


with plt.style.context('dark_background'):
    
# Sujeto de control    
    fig = plt.figure(figsize=(20,4))
    ax1 = fig.add_subplot(121)
    Pxx, freq, t, im = specgram(senial_indep_2500_muestras,NFFT=NFFT,Fs=500,noverlap=noverlap,scale_by_freq='magnitude',detrend=mlab.detrend_linear,window=mlab.window_hanning)
    
    # Plot Spectrogram
    im1 = NonUniformImage(ax1, interpolation='bilinear',extent=(min(t),max(t),10,55),cmap='jet')
    im1.set_data(t,freq,Pxx)
    im1.set_clim()
    ax1.images.append(im1)
    cbar1 = fig.colorbar(im1)
    plt.yscale('linear')
    plt.ylabel('Frequency [Hz]',fontsize=18)
    #xlab = 'Time [seconds] from ' + str(startutc) + ' UTC'
    plt.xlabel('Time',fontsize=18)
    plt.title('Espectrograma sujeto control',fontsize=18)
    plt.xlim(0,max(t))
    plt.ylim(0,36)


# Sujeto dislexia
    
    ax1 = fig.add_subplot(122)
    Pxx_d, freq_d, t_d, im_d = specgram(senial_indep_2500_muestras_dislexia,NFFT=NFFT,Fs=500,noverlap=noverlap,scale_by_freq='magnitude',detrend=mlab.detrend_linear,window=mlab.window_hanning)
    
    # Plot Spectrogram
    im1_d = NonUniformImage(ax1, interpolation='bilinear',extent=(min(t_d),max(t_d),10,55),cmap='jet')
    im1_d.set_data(t_d,freq_d,Pxx_d)
    im1_d.set_clim()
    ax1.images.append(im1_d)
    cbar2 = fig.colorbar(im1_d)
    plt.yscale('linear')
    plt.ylabel('Frequency [Hz]',fontsize=18)
    #xlab = 'Time [seconds] from ' + str(startutc) + ' UTC'
    plt.xlabel('Time',fontsize=18)
    plt.title('Espectrograma sujeto dislexia',fontsize=18)
    plt.xlim(0,max(t_d))
    plt.ylim(0,36)



#Queda añadir el specgram de cada IMF, empiezo por el sujeto normal..
    

#fig_d = plt.figure(figsize=(20,4))
#ax2 = fig_d.add_subplot(121)
Pxx_imf_a = np.ones((5,251,41))
freq_imf_a = np.ones((5,251,41))
t_imf_a = np.ones((5,251,41))
Pxx_imf_norm=[]
freq_imf_norm=[]
t_imf_norm=[]
Pxx_imf_dislexia=[]
im_imf_norm=[]

for i in range(5):
    
#    with plt.style.context('dark_background'):
#        
        #Sujeto control        
        fig3 = plt.figure(figsize=(20,4))
        ax3 = fig3.add_subplot(121)
        
        
        Pxx_imf, freq_imf, t_imf, im_imf = specgram(imfs[i],NFFT=NFFT,Fs=500,noverlap=noverlap,scale_by_freq='magnitude',detrend=mlab.detrend_linear,window=mlab.window_hanning)    
        
        # Plot Spectrogram
        im_imf = NonUniformImage(ax3, interpolation='bilinear',extent=(min(t_imf),max(t_imf),10,55),cmap='jet')
        im_imf.set_data(t_imf,freq_imf,Pxx_imf)
        im_imf.set_clim(0,32)
        
        ax3.images.append(im_imf)
        cbar2 = fig3.colorbar(im_imf)
        
        plt.yscale('linear')
        plt.ylabel('Frequency [Hz]',fontsize=18)
        plt.xlabel('Time',fontsize=18)
        plt.title("IMF " + str(i+1) + " Sujeto de control" ,fontsize=18)
        plt.xlim(0,max(t_imf))
        plt.ylim(0,35)
        
        #Sujeto dislexia
        ax3 = fig3.add_subplot(122)
        Pxx_imf_dislex, freq_imf_dislex, t_imf_dislex, im_imf_dislex = specgram(imfs_dis[i],NFFT=NFFT,Fs=500,noverlap=noverlap,scale_by_freq='magnitude',detrend=mlab.detrend_linear,window=mlab.window_hanning)
    
         
        im_imf_dislex = NonUniformImage(ax3, interpolation='bilinear',extent=(min(t_imf_dislex),max(t_imf_dislex),10,55),cmap='jet')
        im_imf_dislex.set_data(t_imf_dislex,freq_imf_dislex,Pxx_imf_dislex)
        im_imf_dislex.set_clim(0,32)
    
        ax3.images.append(im_imf_dislex)
        cbar2 = fig3.colorbar(im_imf_dislex)
        
        plt.yscale('linear')
        plt.ylabel('Frequency [Hz]',fontsize=18)
        plt.xlabel('Time',fontsize=18)
        plt.title("IMF " + str(i+1) + " Sujeto Dislexia" ,fontsize=18)
        plt.xlim(0,max(t_imf_dislex))
        plt.ylim(0,35)


"""Hasta aquí el cálculo de los descriptores estadísiticos de la señal, filtrado,
espectrogramas, EMD e IMFS junto a todos sus plots"""
    
    

