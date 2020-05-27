# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:36:14 2019

@author: Innu
"""
#Librerias necesarias (no todas)
#import PyEMD 

from PyEMD import EMD
import pandas as pd #librería lectura de datos o frames (.npy y .csv) 
from scipy import signal, interpolate
from scipy.signal import butter, lfilter
import math as m
import numpy as np #librería matemática de py
import pandas as pd #librería lectura de datos o frames (.npy y .csv) 
import matplotlib.pyplot as plt
#import seaborn as sns

from pylab import savefig, specgram
import sys, string 
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import matplotlib.mlab as mlab
from os import makedirs, path
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import tfr_multitaper
import mne


"""Definición de funciones a usar"""

#CORRELACION ENTRE MATRICES


from scipy.spatial.distance import pdist, squareform

def dcov(X, Y):
    """Computes the distance covariance between matrices X and Y.
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov


def dvar(X):
    """Computes the distance variance of a matrix X.
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))


def cent_dist(X):
    """Computes the pairwise euclidean distance between rows of X and centers
     each cell of the distance matrix with row mean, column mean, and grand mean.
    """
    M = squareform(pdist(X))    # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM

def dcor(X, Y):

    assert X.shape[0] == Y.shape[0]

    A = cent_dist(X)
    B = cent_dist(Y)

    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

    return dcor

"""Filtrado de señal, definiciones"""





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


"""Calculo del espectrograma funcion"""


def spectrogram_multitaper(dat,fs):
      
      dat=dat.reshape([1,1,2500])  # Canal, segmento, muestras
      n_epochs=len(dat)
      montage = mne.channels.read_montage('standard_1020',ch_names=['Fp1'])
      info = mne.create_info(['Fp1'],fs, montage=montage,ch_types='eeg')
      #raw=mne.io.RawArray(eegdata,info)
      events = np.array([np.arange(n_epochs), np.ones(n_epochs), np.ones(n_epochs)]).T.astype(int)
      # Construct Epochs. #  eegdata:  epochs, channels, samples
      event_id, tmin, tmax = 1, 0, 5
      #eegdata=eegdata[np.newaxis,:,:]
      epochs = mne.EpochsArray(dat, info, None, 0.0, event_id)
      
      freqs = np.logspace(*np.log10([0.5, 50]), num=100)
      n_cycles = freqs /1.  # different number of cycle per frequency
      time_bandwidth=4.0
      #power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=True, n_jobs=1)
      #power.plot([0], baseline=(-0.5, 0), mode='mean', title=power.ch_names[0],colorbar=True)
      power,itc = tfr_multitaper(epochs, freqs=freqs,n_cycles=n_cycles, time_bandwidth = time_bandwidth,return_itc=True, n_jobs=1)
      #power.plot([0], baseline=(-0.5, 0), mode='mean', title=power.ch_names[0],colorbar=True)
      return power



"""Definicion de la función del cálculo de los descriptores estadísiticos"""

def descpriptor_estad(data,sf):
    
    """Descriptores estad en tiempo"""
    #Cálculo de la media
    ut = data.mean() #media
    
    #Desviación típica: 
    momento_central = data-ut

    momento_central_orden_dos = pow(momento_central,2)
    
    media_ot = momento_central_orden_dos.mean()
    
    ot = m.sqrt(media_ot) #Desviación típica
    
    # Skewness

    aux_1 = momento_central/ot 
    
    Bt = pow(aux_1,3).mean()
    
    """Descriptores estad en freq"""
    #Calculo la power spectra density
    win = 1.25*sf # el tamaño de la ventana es cuantas veces quiero leer la muestra
    freqs, psd = signal.welch(data, sf, nperseg=win)
    
    # Centroide espectral
    
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
    
    return F



#Abro el fichero donde tengo guardado los segmentos
with open('info_segments_2.csv') as file:
    datos = pd.read_csv(file, delimiter= ',', header=0) #leo el fichero y vuelco en datos el valor

#Lo que me interesa a continuacion es separar los sujetos, los de control con los de dislexia
    
#asigno a cada variable la columna correspondientre de datos así separo el csv en columnas independientes
posicion=datos['Unnamed: 0']
identificador=datos['idx']
edad=datos['Age']
source=datos['Source']
dx=datos['DX']
sid=datos['SID']
tipo=datos['Type']
    

"""Cargamos el data set de datos"""
data_signals = np.load('segments_2.npy')


#Separacion de los sujetos del csv
    
comp1 = sid[0] #comp1 y comp2 son variables de comparacione dentro del bucle 
comp2=comp1
num_sujetos=0
sujetos = [] #variable tipo lista para añadir el nombre del sujeto
sujetos.append(comp1)
indice_cambio = [] #var tipo lista para añadir en que posicion cambian los sujetos

ind_usuario_normal = []
ind_usuario_dislex = []

indice_cambio.append(0)
for index, linea in datos.iterrows():
    comp1=linea['SID']
    if(comp1 != comp2):
        if(linea['DX'] == 'c'):
            ind_usuario_normal.append(index)
        else:
            ind_usuario_dislex.append(index)
        
        num_sujetos=num_sujetos+1
        sujetos.append(comp1)
        indice_cambio.append(index)
    comp2=comp1
indice_cambio.append(index)
if(linea['DX'] == 'c'):
    ind_usuario_normal.append(index)
else:
    ind_usuario_dislex.append(index)    
""" En esta parte lo que hago es buscar tanto el numero de sujetos que tengo
como saber quienes son esos sujetos y en que posicion cambian. Para que al asociar 
señal a sujeto, oredenarlas por sujeto y asociar señal a sujeto sin equivocarnos """
    





#Ahora pasaremos a normalizar cada segmento, para ello se emplea la sklearn

scaler = StandardScaler()

data = []

for i in range(index):
#    scaler.fit(data_signals[:,i,:])
    aux = data_signals[:,i,:]
    data.append(aux)
    
data_nomr = np.array(data)
    
    
#Ya tenemos los segmentos normalizados ahora hay que crear el segmento medio 
# por cada sujeto

#Segmentos medios
data_final = []
for n in range(num_sujetos+1): #Debido a que hay 50 sujetos y yo empiezo contando en num_sujetos = 0
    aux1 = np.mean(data_nomr[indice_cambio[n]:indice_cambio[n+1],:,:],axis=0)
    aux1 = scaler.fit_transform(aux1)
    data_final.append(aux1)
 
data_final = np.array(data_final)

#Separamos sujetos los de control con los de dislexia 
data_control = []
data_dislexia = []

#Observando la variacion de los íncices, vemos que claramente el data set está dividido hasta el 1055
#en usuarios de control y hasta el 1527 en usuarios dislexicos

#Por lo que tendríamos 33 usuarios de control y 17 dislexicos

for k in range(num_sujetos+1):
    if k<33:
        data_control.append(data_final[k,:,:])
    else: 
        data_dislexia.append(data_final[k,:,:])
        
        
data_control = np.array(data_control)
data_dislexia = np.array(data_dislexia)

"""Ahora paso a la parte del filtrado, ya tenemos los segmentos medios de cada tipo de sujeto"""

"""Filtrado de las señales medias"""
#Sujeto normal
y_filtrada = butter_lowpass_filter(data_control, cutoff_l, fs, order)
y_filtrada = butter_highpass_filter(y_filtrada, cutoff_h, fs, order)
control = y_filtrada

#Sujeto dislexia
y_filtrada_dislex = butter_lowpass_filter(data_dislexia, cutoff_l, fs, order)
y_filtrada_dislex = butter_highpass_filter(y_filtrada_dislex, cutoff_h, fs, order)
dislexia = y_filtrada_dislex

#Pasamos a la descomposicion EMD para cada suejeto

"""Descomposicion EMD de la señal"""
print("Performing decomposition... ")
secsPerFFT = 2 # Hz
overlap = 0.5 # fractional overlap
Fs = 500
NFFT = int(round(Fs*secsPerFFT))
noverlap = int(round(overlap*NFFT))


Sujetos_control_0 = []
Sujetos_control_1 = []
Sujetos_control_2 = []
Sujetos_control_3 = []
Sujetos_control_4 = []

Sujetos_dislexia_0 = []
Sujetos_dislexia_1 = []
Sujetos_dislexia_2 = []
Sujetos_dislexia_3 = []
Sujetos_dislexia_4 = []

for suj_ctrl in range(33): #33 sujetos control
    for s in range(32):
        aux = control[suj_ctrl,s,:]
        emd = EMD()
        emd.emd(aux, max_imf=5)
    
        imfs,res = emd.get_imfs_and_residue()
        
        print("CONTROL CANAL " +str(s+1))
        for b in range (5):     
 
            desp = descpriptor_estad(imfs[b],fs)
            
            if b == 0:
                
                descriptor_control_0 = desp
            elif b==1:
                
                descriptor_control_1 = desp
            elif b==2:
                
                descriptor_control_2 = desp
            elif b==3:
                
                descriptor_control_3 = desp
            elif b==4:
                
                descriptor_control_4 = desp
                
        Sujetos_control_0.append(descriptor_control_0)
        Sujetos_control_1.append(descriptor_control_1)
        Sujetos_control_2.append(descriptor_control_2)
        Sujetos_control_3.append(descriptor_control_3)
        Sujetos_control_4.append(descriptor_control_4)  
        
    print("VUELTA : " + str(suj_ctrl+1))


Sujetos_control_0 = np.array(Sujetos_control_0)
Sujetos_control_1 = np.array(Sujetos_control_1)
Sujetos_control_2 = np.array(Sujetos_control_2)
Sujetos_control_3 = np.array(Sujetos_control_3)
Sujetos_control_4 = np.array(Sujetos_control_4)
          
Sujetos_control_0 = np.split(Sujetos_control_0,32)
Sujetos_control_1 = np.split(Sujetos_control_1,32) 
Sujetos_control_2 = np.split(Sujetos_control_2,32) 
Sujetos_control_3 = np.split(Sujetos_control_3,32) 
Sujetos_control_4 = np.split(Sujetos_control_4,32)     
    
    
Suj_Control = np.array([Sujetos_control_0,Sujetos_control_1,Sujetos_control_2,Sujetos_control_3,Sujetos_control_4])    
#Sujeto Dislexia    
for suj_dis in range(17): #33 sujetos control
    for s_d in range(32):
        aux = dislexia[suj_dis,s_d,:]
        emd = EMD()
        emd.emd(aux, max_imf=5)
    
        imfs,res = emd.get_imfs_and_residue()
        
        
        
        
        print("DISLEXIA CANAL " +str(s_d+1))
        for b_d in range (5):     
 
            
            desp = descpriptor_estad(imfs[b_d],fs)
            
            if b_d == 0:
                
                descriptor_dislexia_0 = desp
            elif b_d==1:
                
                descriptor_dislexia_1 = desp
            elif b_d==2:
                
                descriptor_dislexia_2 = desp
            elif b_d==3:
                
                descriptor_dislexia_3 = desp
            elif b_d==4:
               
                descriptor_dislexia_4 = desp

    
        Sujetos_dislexia_0.append(descriptor_dislexia_0)
        Sujetos_dislexia_1.append(descriptor_dislexia_1)
        Sujetos_dislexia_2.append(descriptor_dislexia_2)
        Sujetos_dislexia_3.append(descriptor_dislexia_3)
        Sujetos_dislexia_4.append(descriptor_dislexia_4)

    print("VUELTA : " + str(suj_dis+1)) 

Sujetos_dislexia_0 = np.array(Sujetos_dislexia_0)
Sujetos_dislexia_1 = np.array(Sujetos_dislexia_1)
Sujetos_dislexia_2 = np.array(Sujetos_dislexia_2)
Sujetos_dislexia_3 = np.array(Sujetos_dislexia_3)
Sujetos_dislexia_4 = np.array(Sujetos_dislexia_4)
    
            
Sujetos_dislexia_0 = np.split(Sujetos_dislexia_0,32)
Sujetos_dislexia_1 = np.split(Sujetos_dislexia_1,32)
Sujetos_dislexia_2 = np.split(Sujetos_dislexia_2,32)
Sujetos_dislexia_3 = np.split(Sujetos_dislexia_3,32)
Sujetos_dislexia_4 = np.split(Sujetos_dislexia_4,32)       


Suj_Dislexia = np.array([Sujetos_dislexia_0,Sujetos_dislexia_1,Sujetos_dislexia_2,Sujetos_dislexia_3,Sujetos_dislexia_4])

#Guardaremos ambos tipos de sujeto al estar ordenados en una matriz conjunta de sujetos para que a 
#continuació se implemente el vector columna que necesito para la identificacion de los suejtos


#Matriz_sujetos = np.array(Matriz_sujetos)

Matriz_sujetos = np.zeros((5,32,50,6))
Tipo_sujetos = np.zeros((50,1))
q = 0

for sjtos in range(50):
    if(sjtos<33):
        #Control 
        Tipo_sujetos[sjtos] = 1
        Matriz_sujetos[:,:,sjtos,:] = Suj_Control[:,:,sjtos,:]
    elif(sjtos>=33):
        #Dislexia
        
        Tipo_sujetos[sjtos] = -1
        Matriz_sujetos[:,:,sjtos,:] = Suj_Dislexia[:,:,q,:]
        q+=1
        
print("FIN Proceso I")

M_Sujetos_final = np.zeros((50,960))
M_aux = []
for rango in range(50):
    M_aux.clear()
    for imfss in range(5):
        for chanel in range(32):
            for carac in range(6):
                M_aux.append(Matriz_sujetos[imfss,chanel,rango,carac])
    aux_h = np.array(M_aux)
        
    M_Sujetos_final[rango] = aux_h
    
    
#M_Sujetos_final = np.array(M_Sujetos_final)

np.savez('Caracteristicas.npz',M_Sujetos_final=M_Sujetos_final, Tipo_sujetos=Tipo_sujetos)

data=np.load('Caracteristicas.npz')
M_Sujetos_final=data['M_Sujetos_final']
Tipo_sujetos=data['Tipo_sujetos']




#Data_pca = scaler.fit_transform(M_Sujetos_final)

from sklearn.decomposition import PCA#, RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
#from sklearn.datasets import make_classification
# Instansiate normalized (standard)
scaler=StandardScaler()
#Instansiate classifier



def pca(data, npc, plot=False):
      #normalizamos los datos
#      scaler.fit(data) # calculo la media para poder hacer la transformacion
#      data_norm=scaler.transform(data)# Ahora si, escalo los datos y los normalizo
      data_norm = data
      #Instanciamos objeto PCA y aplicamos
      pca=PCA(n_components=npc) 
      # Otra opción es instanciar pca sólo con dimensiones nuevas hasta obtener un mínimo "explicado"       ej.: pca=PCA(.85)
      #pca=RandomizedPCA(n_components=9)
      pca.fit(data_norm) # obtener los componentes principales
      data_proj=pca.transform(data_norm) # convertimos nuestros datos con las nuevas dimensiones de PCA
      # Varianza explicada
      expl = pca.explained_variance_ratio_
      #print(expl)
      #print('suma:',sum(expl[0:5]))
      if plot:
            #graficamos el acumulado de varianza explicada en las nuevas dimensiones
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.show()
      return scaler, pca, data_proj, expl

Data_pca = M_Sujetos_final



#Data_Proj_norm = scaler.fit_transform(Data_pca)

Scal_p, Red_pca_p, Data_Proj_p, Expl_p = pca(Data_pca,19,True)
Data_Proj_norm = scaler.fit_transform(Data_Proj_p)
plt.grid()
plt.xlabel("NºComponentes", size = 25)
plt.ylabel("Varianza explicada", size = 25)
plt.title("PCA", size = 25)
plt.xlim(0,50)
plt.plot(46,0.9999999999,'ro')
plt.annotate("99% varianza explicada\n Nºcomponentes = 46", (38.24,0.86), size=30)
plt.plot(19.11,0.849,'ro')
plt.annotate("85% varianza explicada\n Nºcomponentes = 19", (11.02,0.906), size=30)


#x_scat = Data_Proj_p[:,0]
#y_scat = Data_Proj_p[:,1]
#
#plt.scatter(x_scat,y_scat)



def performance_metrics(y_true, y_pred):
      from sklearn.metrics import accuracy_score
      from sklearn.metrics import precision_score
      from sklearn.metrics import recall_score
      from sklearn.metrics import f1_score
      from sklearn.metrics import cohen_kappa_score
      from sklearn.metrics import roc_auc_score
      from sklearn.metrics import confusion_matrix
      
      # accuracy: (tp + tn) / (p + n)
      accuracy = accuracy_score(y_true, y_pred)
      
      tn, fp , fn, tp = confusion_matrix(y_true,y_pred).ravel()
      
      specificity= tn / (tn+fp)
      sensitivity= tp / (tp+fn)
      
      # ROC AUC
      #auc = roc_auc_score(y_true, y_probs)
      #print('ROC AUC: %f' % auc)
      return accuracy, sensitivity, specificity






""" Parte asociada al ML y clasificación de los datos """

acc=0
sns=0
esp=0
ac_rep = []
sen = []
espf = []
resultados = []
margenes = []
from sklearn.svm import OneClassSVM 
Fmat = Data_Proj_norm
lab = Tipo_sujetos
lab=lab.flatten()

kernel = ['linear', 'poly', 'rbf' , 'sigmoid']
n_splits=5
nu = [0.1,0.25,0.5]

for nucleo in range(len(kernel)):
    for n in range(len(nu)):
        kf = StratifiedKFold(n_splits)
        clf = OneClassSVM(nu=nu[n], kernel=kernel[nucleo], gamma='auto')
        
        
        for train_index, test_index in kf.split(Fmat,lab):  #Validación cruzada
            
            trainlabels=lab[train_index].flatten()
            testlabels=lab[test_index].flatten()
            
            train_data=scaler.fit_transform(Fmat[train_index,:])
            test_data=scaler.transform(Fmat[test_index,:])
            
            CN=train_data[trainlabels==1,:]
            CN=np.vstack((CN,test_data[testlabels==1]))
            
            clf.fit(CN)
            pred=clf.predict(test_data)
            prob=clf.decision_function(test_data)
            
            accuracy, sensitivity, specifity = performance_metrics(testlabels,pred)
            #print("acc=", accuracy, "sens=", sensitivity, "spec=", specifity)
        
            ac_rep.append(accuracy)
            sen.append(sensitivity)
            espf.append(specifity)
            
            acc+=accuracy
            sns+=sensitivity
            esp+=specifity
        
        std_a = round(np.std(ac_rep),3)
        std_s = round(np.std(sen),3)
        std_e = round(np.std(espf),3)
        
        acc_tot = round(acc/n_splits,3)
        sens_tot = round(sns/n_splits,3)
        esp_tot = round(esp/n_splits,3)
        R = [kernel[nucleo], nu[n], acc_tot, std_a, sens_tot, std_s, esp_tot, std_e]
        m = [R[2]+R[3], R[2]-R[3], R[4]+R[5], R[4]-R[5], R[6]+R[7], R[6]-R[7] ]
        margenes.append(m)
        resultados.append(R) #Resultados por cada tipo de núcleo y nu tras modificar el númeor de kfold (kfold se modifica a mano para cada numero de splits deseados)
        acc=0
        sns=0
        esp=0
        ac_rep.clear()
        sen.clear()
        espf.clear()
        
        print("acc total=", acc_tot, "sns_total=", sens_tot, "specif tot=", esp_tot)
        
        

ac_t = []
se_t = []
es_t = []




# Gráfico realizado para comparar la acc, sens y espef por cada componente hasta 50.

for comp in range(50):
    
    Scal_p, Red_pca_p, Data_Proj_p, Expl_p = pca(Data_pca,comp+1,False)
    Data_Proj_norm = scaler.fit_transform(Data_Proj_p)
    
    acc=0
    sns=0
    esp=0
    ac_rep = []
    sen = []
    espf = []
    Fmat = Data_Proj_norm
    lab = Tipo_sujetos
    lab=lab.flatten()
    
    n_splits=5
    
    
    
    kf = StratifiedKFold(n_splits)
    clf = OneClassSVM(nu=0.5, kernel='rbf', gamma='auto')
    
    
    for train_index, test_index in kf.split(Fmat,lab):
        
        trainlabels=lab[train_index].flatten()
        testlabels=lab[test_index].flatten()
        
        train_data=scaler.fit_transform(Fmat[train_index,:])
        test_data=scaler.transform(Fmat[test_index,:])
        
        CN=train_data[trainlabels==1,:]
        CN=np.vstack((CN,test_data[testlabels==1]))
        
        clf.fit(CN)
        pred=clf.predict(test_data)
        prob=clf.decision_function(test_data)
        
        accuracy, sensitivity, specifity = performance_metrics(testlabels,pred)
        #print("acc=", accuracy, "sens=", sensitivity, "spec=", specifity)
    
        ac_rep.append(accuracy)
        sen.append(sensitivity)
        espf.append(specifity)
        
        acc+=accuracy
        sns+=sensitivity
        esp+=specifity
    
    std_a = np.std(ac_rep)
    std_s = np.std(sen)
    std_e = np.std(espf)
    
    acc_tot = acc/n_splits
    sens_tot = sns/n_splits
    esp_tot = esp/n_splits
    
    ac_t.append(acc_tot)
    se_t.append(sens_tot)
    es_t.append(esp_tot)
    
    acc=0
    sns=0
    esp=0
    ac_rep.clear()
    sen.clear()
    espf.clear()
        
       
plt.grid()
plt.xlabel("NºComponentes", size = 25)
plt.ylabel("Accuracy, Sensibilidad, Especificidad", size = 25)
plt.title("Nº componentes vs parámetros empelados para el diagnóstico", size = 25)
plt.xlim(0,50)
plt.plot(ac_t, 'r', label = "Accuracy")
plt.plot(se_t, 'g', label = "Sensibilidad")
plt.plot(es_t, 'y', label = "Especificidad")
plt.legend()        
    
    
