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

"""
threshold=[]
with open("data/polymer-ae/feature/thresold.csv", 'r') as file:
          csvreader = csv.reader(file)
          for row in csvreader:
            threshold.append(row[1])
###print(threshold)
"""

featuresName = []
AEfeatures = []

def featuresCalculation(thresholdedDataArr):
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
print(featuresName, AEfeatures)

def readAndProcessDataForFeatures(directoryPath):
    


    
    for root, dirs, files in os.walk(directoryPath):
        for thresholdedDirName in dirs:
            #print(dirs)
            
        

            txtFilesLocation = glob.glob(root+thresholdedDirName+'/*.txt')
            for fileLocation in txtFilesLocation:
                ##print(xtFilesLocation)

                
                thresholdedDataDf = np.loadtxt(fileLocation)
                ##print(thresholdedDataDf)



                ##featuresName, AEfeatures = featuresCalculation(thresholdedDataDf)



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

                ##print((dataDir/'feature').exists())
                if ((dataDir/'feature').exists()):
                    featureDir = str(dataDir/'AEfeatures')
                    if (Path(featureDir + '/' + thresholdedDirName).exists()):
                        if(Path(featureDir + '/' + thresholdedDirName + '/AEfeatures.csv').exists()):
                            if (appendMode):
                                # Convert the array to a space-separated string
                                row_string = ' '.join(str(x) for x in AEfeatures)
                                # Split the string on spaces to create a list of values
                                values = row_string.split()
                                # Write the values to the CSV file                            
                                with open(Path(featureDir + '/' + thresholdedDirName + '/AEfeatures.csv'), 'a', newline='') as file:
                                    writer = csv.writer(file)                                
                                    writer.writerow(values)
                            else:
                                # replace the feature for same experiment, yet to complete the mode
                                experimentType = AEfeaturesDf['Experiment_type'].values
                                if (thresholdedDirName in experimentType):
                                    
                                    index = AEfeaturesDf.index[AEfeaturesDf['Experiment_type'] == thresholdedDirName].tolist()
                                    
                                    AEfeaturesDf.iloc[index[0]] = AEfeatures
                                    featureFile = dataDir + '/polymer-ae/feature/AEfeatures.csv'
                                    #featureDf.to_csv(dataDir/'feature/feature.csv', index = False)
                        else:
                            AEfeaturesDf = pd.DataFrame([AEfeatures], columns = featuresName)
                            #Save the feature data frame
                            featureFile = featureDir + "/" + thresholdedDirName + '/features.csv'
                            AEfeaturesDf.to_csv(featureFile, index = False) 

                    else:
                        print('expt directory not exist')
                        mode = 0o777
                        path = os.path.join(featureDir, thresholdedDirName)
                        os.mkdir(path, mode)        
                        print(' exp directory is created')
                    
                        #Save the denoised data to a new CSV file
                        AEfeaturesDf = pd.DataFrame([AEfeatures], columns = featuresName)
                        #Save the feature data frame
                        featureFile = featureDir + "/" + thresholdedDirName + '/AEfeatures.csv'
                        AEfeaturesDf.to_csv(featureFile, index = False)

                else:
                    # Create feature directory
                    mode = 0o777
                    path = os.path.join(dataDir, 'AEfeatures')
                    os.mkdir(path, mode)
                    print('"features" directory is created')
                    featureDir = str(dataDir/'AEfeatures')
                    
                    # Create experiment-type directory withing the feature directory
                    path = os.path.join(featureDir, thresholdedDirName)
                    os.mkdir(path, mode)        
                    print(thresholdedDirName + ' Directory is created')

                    AEfeaturesDf = pd.DataFrame([AEfeatures], columns = featuresName)
                    #Save the feature data frame
                    featureFile = featureDir + "/" + thresholdedDirName + '/features.csv'
                    AEfeaturesDf.to_csv(featureFile, index = False)



directoryPath = os.getcwd()+'/data/polymer-ae/thresholded/'
readAndProcessDataForFeatures(directoryPath)
print("directoryPath")
