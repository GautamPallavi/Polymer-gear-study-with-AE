import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import glob
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, entropy
import csv
from src.exception import CustomException



@dataclass
class DataDenoiseConfig:
    train_data_path:str=os.path.join('artifacts',"denoise_data.csv")

class DataDenoise:
    def __init__(self):
        self.data_Denoise_config=DataDenoiseConfig()
    def __init__(self):
        self.madev(d, axis=None)
    """ Mean absolute deviation of a signal """
    

    def get_data_Denoiser_object(self):

        self.wavelet = 'sym2'
        self.level = 5
        self.mode = 'periodic'
    

        try:
            def madev(d, axis=None):
    coeff = pywt.wavedec(rawDataNdar, wavelet, mode)
    sigma = (1/0.6745) * madev(coeff[-level])*2
    uthresh = sigma * np.sqrt(2 * np.log(len(rawDataNdar)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    denoisedDataArr = pywt.waverec(coeff, wavelet, mode)

        


    #rawDataNdar = rawDataNdar[0:10000]
    #print(data)

    
    '''  
    if (plogGraph):
        plt.figure(figsize=(10, 6))
        plt.plot(rawDataNdar, label = 'Raw signal')
        plt.plot(denoisedDataArr, label = 'Denosied signal')
        plt.xlabel('samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title(f"DWT Denoising with {wavelet} Wavelet", size=10)
        plt.show()
    '''




