# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:53:18 2019

@author: Innu
"""
import matplotlib.pyplot as plt
import math as m
import numpy as np #librería matemática de py
import pandas as pd #librería lectura de datos o frames (.npy y .csv) 
import seaborn as sns

# Abro el fichero donde tengo guardado os segmentos
with open('info_segments_2.csv') as file:
    datos = pd.read_csv(file, delimiter= ',', header=0) #leo el fichero y vuelco en datos el valor
    

data_signals = np.load('segments_2.npy') #cargo las señales en data 


#------------ESTADÍSTICAS TEMPORALES (C)-----------------------#

# Voy a trabajar primero con las 2500 muestras de un sujeto

senial_indep_2500_muestras = data_signals[5,2,:]



# Una vez extraída la señal hago la media: 

ut = senial_indep_2500_muestras.mean() #media
ut_prima = sum(senial_indep_2500_muestras)/2500 #obteniendo la media de otra manera también

# Desviación típica: 

# Preparo el momento central de orden 2 

momento_central = senial_indep_2500_muestras-ut

momento_central_orden_dos = pow(momento_central,2)

media_ot = momento_central_orden_dos.mean()

ot = m.sqrt(media_ot) # Desviación típica



# Skewness

aux_1 = momento_central/ot 

Bt = pow(aux_1,3).mean()


""" Hasta aquí el análisis en tiempo, a continuación pinto la gráfica de la señal """


sns.set(font_scale=1.2)

# Define sampling frequency and time vector (Reduce la escala del muestreo)
sf = 500. #Según valga sf puedo hacer más pequeña o no mi ventana
# Mi ventana es de 500 debido a que tengo en 5 segundos 2500 muestras

""" Fs = N/t así que como N = 2500 y t = 5s sale una freq de muestreo de 500 Hz -> es sf"""
time = np.arange(senial_indep_2500_muestras.size)/sf

# Dibujo la señal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, senial_indep_2500_muestras, lw=1.5, color='k')
plt.xlabel('Tiempo')
plt.ylabel('Valor de la muestra (uV)')
plt.xlim([time.min(), time.max()])
plt.title('Onda generada por un electrodo')
sns.despine()


#-----------ANÁLISIS ESPECTRAL (D)-------------------------#

"""Debido a que en un amplio rango de señal es muy dificil que la señal sea una suma
periódica sinusoidal se emplea el método de Weltch ya que trabaja con pequeños trozos de señal
(ventanas) para calcular el espectro"""

from scipy import signal

win = 1.25*sf # el tamaño de la ventana es cuantas veces quiero leer la muestra
freqs, psd = signal.welch(senial_indep_2500_muestras, sf, nperseg=win)

"""The freqs vector contains the x-axis (frequency bins) and the psd vector contains the y-axis (power spectral density). 
The units of the power spectral density, when working with EEG data, is usually micro-Volts-squared per Hz """

# Plot the power spectrum (densidad) -> P(w)
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, freqs.max()])
sns.despine()




# Centroide espectral:
w = freqs*2*m.pi
aux_2 = psd*w

Cs = sum(aux_2)/sum(psd) #centroide espectral

# Coeficiente de variacion espectral

aux_3 = w-Cs
aux_4 = pow(aux_3,2)
aux_5 = psd*aux_4
os_cuadrado = sum(aux_5)/sum(psd)

os = m.sqrt(os_cuadrado)



# Spectral Skew

aux_6 = pow(aux_3/os,3)*psd

Bs = sum(aux_6)/sum(psd)


# Por último concatenamos estos resultados en un único vector F para su estudio:
 
F = [ut, ot, Bt, Cs, os, Bs]

sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(F, color='k', lw=2)
plt.plot(F, 'ro')
plt.xlabel('Caracteríticas')
plt.ylabel('Valor de las caracterísitcas')
plt.title("Sujeto de Control")
plt.xlim([-0.5, 6])
sns.despine()