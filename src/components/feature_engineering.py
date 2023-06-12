import pandas as pd
import numpy as np
import os
import pywt
import glob
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, entropy
import csv
from src.exception import CustomException

#from src.components import data-thresholding
#from logger import logging

def featuresCalculation(thresholdedDataArr):
    featuresName = []
    AEfeatures = []

    # Calculate the statistical features of the AE signal
    amplitude = np.max(thresholdedDataArr) - np.min(thresholdedDataArr)
    featuresName.append('amplitude')
    AEfeatures.append(amplitude)
    
    peakIndices, peakProperties = find_peaks(thresholdedDataArr, distance = 100)
    featuresName.append('peaks')
    AEfeatures.append(len(peakIndices))
    print(peakProperties)
    
    rise_time = abs(np.argmax(thresholdedDataArr) - np.argmin(thresholdedDataArr))
    featuresName.append('rise_time')
    AEfeatures.append(rise_time)
    
    duration = np.sum(thresholdedDataArr > 0.1*np.max(thresholdedDataArr))
    featuresName.append('duration')
    AEfeatures.append(duration)
     
    counts = np.sum(thresholdedDataArr > 0)
    featuresName.append('counts')
    AEfeatures.append(counts)
    
    energy = np.sum(np.square(thresholdedDataArr))
    featuresName.append('energy')
    AEfeatures.append(energy)
    
    peak_amplitude = np.argmax(thresholdedDataArr)
    featuresName.append('time_to_peak')
    AEfeatures.append(peak_amplitude)
    
    #Calculate the skewness and kurtosis of the AE signal
    skewness = skew(thresholdedDataArr)
    featuresName.append('skewness')
    AEfeatures.append(skewness)
    
    kurt = kurtosis(thresholdedDataArr)
    featuresName.append('kurt')
    AEfeatures.append(kurt)

    ae_rms=np.sqrt(np.mean(thresholdedDataArr**2))
    featuresName.append('ae_rms')
    AEfeatures.append(ae_rms)


    #return (amplitude, rise_time, duration, count, energy, freq, peak_amplitude, skewness, kurt)
    #print(featuresName, AEfeatures)
    return featuresName, AEfeatures

def readAndProcessDataForFeatures(directoryPath):
    
    for root, dirs, files in os.walk(directoryPath):
        for thresholdedDirName in dirs:

            txtFilesLocation = glob.glob(root+thresholdedDirName+'/*.txt')
            for fileLocation in txtFilesLocation:
                ##print(xtFilesLocation)

                thresholdedDataDf = np.loadtxt(fileLocation)
                ##print(thresholdedDataDf)

                featuresName, AEfeatures = featuresCalculation(thresholdedDataDf)

                #Get the file to extract the load and speed label for feature

                fileForFeatureLabel = str(fileLocation)
                fileForFeatureLabel = fileForFeatureLabel.replace(root+thresholdedDirName + '/', '')               
                featuresName.insert(0, 'speed')
                AEfeatures.insert(0, fileForFeatureLabel[0:2])
                featuresName.insert(0, 'load')
                AEfeatures.insert(0, fileForFeatureLabel[5:6])

                featuresName.insert(0, 'experiment_type')
                AEfeatures.insert(0, thresholdedDirName)
                dataDir= Path(root).parent

                featureDir = str(dataDir/'AEfeatures')
                featuresFile = featureDir + "/features.csv"

                featuresName = np.array(featuresName)
                #AEfeatures = np.array(AEfeatures)
                AEfeaturesNpArr = np.array([AEfeatures])
            
                #--- Check AEfeatures folder exist or not
                if ((dataDir/'AEfeatures').exists()):

                    #check threshold file exist or not
                    if (Path(featuresFile).exists()):
                        #--- Append values
                        with open(featuresFile, 'a') as csv_file:
                            dict_object = csv.writer(csv_file, delimiter=',')
                            dict_object.writerow(AEfeatures)
                            csv_file.close()
                    else:
                        np.savetxt(featuresFile,AEfeaturesNpArr, delimiter=",", fmt='%s')

                else:
                    mode = 0o777
  
                    # Path
                    path = os.path.join(dataDir, 'AEfeatures')
                    
                    os.mkdir(path, mode)
                    featureDir = str(dataDir/'AEfeatures')
                    print('"AEfeatures" directory is created')
                    
                    #check threshold file exist or not
                    if (Path(featuresFile).exists()):
                        #--- Append values
                        with open(featuresFile, 'a') as csv_file:
                            dict_object = csv.writer(csv_file, delimiter=',')
                            dict_object.writerow(AEfeatures)
                            csv_file.close()
                    else:
                        np.savetxt(featuresFile,AEfeaturesNpArr, delimiter=",", fmt='%s')

    print('Features calculated')
    return

directoryPath = os.getcwd()+'/data/polymer-ae/thresholded/'
readAndProcessDataForFeatures(directoryPath)
