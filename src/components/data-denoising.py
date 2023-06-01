import pandas as pd
import numpy as np
import os
import pywt
import glob
from pathlib import Path
#from scipy.signal import find_peaks
#from scipy.stats import skew, kurtosis, entropy
#import csv
from src.exception import CustomException



def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def dataDenoising(rawDataNdar):
    #Get the list of all the avialable wevelet 
    #print(pywt.wavelist(kind='discrete'))
    wavelet = 'sym2'
    level = 5
    mode = 'periodic'

    #rawDataNdar = rawDataNdar[0:10000]
    #print(data)

    coeff = pywt.wavedec(rawDataNdar, wavelet, mode)
    sigma = (1/0.6745) * madev(coeff[-level])*2
    uthresh = sigma * np.sqrt(2 * np.log(len(rawDataNdar)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    denoisedDataArr = pywt.waverec(coeff, wavelet, mode)
    #denoisedData = pd.DataFrame(denoisedData)
    return denoisedDataArr


def readAndProcessDataForDenoising(directoryPath):
    for root, dirs, files in os.walk(directoryPath):
        for rowDirName in dirs:
            #print(rowDirName)
            txtFiles = glob.glob(root+rowDirName+'/*.txt')
            for file in txtFiles:
                print(txtFiles)
                #rawDataDf = pd.read_csv(file)
                #rawDataDf = rawDataDf['AE_data'].values
                rawDataDf = np.loadtxt(file)
                denoisedDataArr = dataDenoising(rawDataDf)
                #denoisedDataDf = pd.DataFrame(denoisedDataArr , columns = ['AE_data'])
                fileName = file.replace(root+rowDirName+'/','')
                
                #assign file name t coulmn name
                #speed = fileName[0:2]
                #load = fileName[5:6]
                #denoisedData.insert(1, 'load', load)
                #denoisedData.insert(2, 'speed', speed)
         
                
                aeExpDir = Path(root).parent

                #check denoised directory exist or not
                if ((aeExpDir/'denoised').exists()):
                    denoisedDir = str(aeExpDir/'denoised')
                    
                    #check internalforlder exist or not
                    if (Path(aeExpDir/'denoised'/rowDirName).exists()):
                           #denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName + '.csv'
                        denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName
                           #denoisedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(denoisedFile,denoisedDataArr, delimiter=",")
                    else:
                        mode = 0o777
                        path = os.path.join(denoisedDir, rowDirName)
                        os.mkdir(path, mode)        
                        print(rowDirName + ' Directory is created')
                          #Save the denoised data to a new CSV file
                          #denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName + '.csv'
                        denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName
                        #denoisedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(denoisedFile, denoisedDataArr, delimiter=",")

                else:
                    mode = 0o777
  
                    # Path
                    path = os.path.join(aeExpDir, 'denoised')
                    
                    os.mkdir(path, mode)
                    denoisedDir = str(aeExpDir/'denoised')
                    print('"denoised" directory is created')
                    
                    #check
                    if (Path(aeExpDir/'denoised'/rowDirName).exists()):
                        #denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName + '.csv'
                        denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName
                        #denoisedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(denoisedFile, denoisedDataArr, delimiter=",")
                    else:
                        mode = 0o777
  
                        # Path
                        path = os.path.join(denoisedDir, rowDirName)
                        os.mkdir(path, mode)        
                        print(rowDirName + ' Directory is created')
                    
                        #Save the denoised data to a new CSV file
                        #denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName + '.csv'
                        denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName
                        #denoisedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(denoisedFile, denoisedDataArr, delimiter=",")
        
    return

directoryPath = os.getcwd()+'/data/polymer-ae/raw/'
readAndProcessDataForDenoising(directoryPath)