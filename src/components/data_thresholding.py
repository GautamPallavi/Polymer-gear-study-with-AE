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
# Import writer class from csv module
from csv import writer
import csv
#from logger import logging



def thresholding(denoisedDataArr):
    threshold = np.mean(denoisedDataArr) + 2.5 * np.std(denoisedDataArr)

    #logging.info("denoised data is passed to data threshold function")

    thresholdedDataArr = np.where(denoisedDataArr > threshold, denoisedDataArr, 0)

    ##peakIndices, peakProperties = find_peaks(thresholdedDataArr, distance = 4000)
    return threshold, thresholdedDataArr


def readAndProcessDataForThresolding(directoryPath):
    for root, dirs, files in os.walk(directoryPath):
        for denoisedDirName in dirs:
            txtFiles = glob.glob(root+denoisedDirName+'/*.txt')
            for file in txtFiles:
                denoisedDataDf = np.loadtxt(file)
                threshold, thresholdedDataArr = thresholding(denoisedDataDf)
                fileName =  file.replace(root+denoisedDirName+'/','')
                aeExpDir = Path(root).parent

                # check if denoised directory exist or not
                if ((aeExpDir/'thresholded').exists()):
                    denoisedDir = str(aeExpDir/'thresholded')

                    #check internalforlder exist or not
                    if (Path(aeExpDir/'thresholded'/denoisedDirName).exists()):
                        thresholdedFile = denoisedDir + "/" + denoisedDirName + "/" + fileName
                        #thresholdedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(thresholdedFile,thresholdedDataArr, delimiter=",")
                    else:
                        mode = 0o777
                        path = os.path.join(denoisedDir, denoisedDirName)
                        os.mkdir(path, mode)        
                        print(denoisedDirName + ' Directory is created')
                    
                        #Save the denoised data to a new CSV file
                        thresholdedFile = denoisedDir + "/" + denoisedDirName + "/" + fileName
                        #thresholdedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(thresholdedFile,thresholdedDataArr, delimiter=",")

                else:
                    mode = 0o777
  
                    # Path
                    path = os.path.join(aeExpDir, 'thresholded')
                    
                    os.mkdir(path, mode)
                    denoisedDir = str(aeExpDir/'thresholded')
                    print('"thresholded" directory is created')
                    
                    #check
                    if (Path(aeExpDir/'thresholded'/denoisedDirName).exists()):
                        thresholdedFile = denoisedDir + "/" + denoisedDirName + "/" + fileName
                        #thresholdedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(thresholdedFile,thresholdedDataArr, delimiter=",")
                    else:
                        mode = 0o777
  
                        # Path
                        path = os.path.join(denoisedDir, denoisedDirName)
                        os.mkdir(path, mode)        
                        print(denoisedDirName + ' Directory is created')
                    
                        #Save the denoised data to a new CSV file
                        thresholdedFile = denoisedDir + "/" + denoisedDirName + "/" + fileName
                        #thresholdedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(thresholdedFile,thresholdedDataArr, delimiter=",")


                #--- Save threshold values in new file in feature folder ----------
                featureDir = str(aeExpDir/'feature')
                thresholdFeatureFile = featureDir + "/" + 'thresold.csv'
                thresholdArrToSave = [fileName, threshold]                    
                thresholValuWithFileName = np.array([thresholdArrToSave])
                


                # check if denoised directory exist or not
                if ((aeExpDir/'feature').exists()):
                    #check threshold file exist or not
                    if (Path(thresholdFeatureFile).exists()):
                        #--- Append values
                        with open(thresholdFeatureFile, 'a') as csv_file:
                            dict_object = csv.writer(csv_file, delimiter=',')
                            dict_object.writerow(thresholdArrToSave)
                            csv_file.close()
                    else:
                        np.savetxt(thresholdFeatureFile,thresholValuWithFileName, delimiter=",", fmt='%s')

                else:
                    mode = 0o777
  
                    # Path
                    path = os.path.join(aeExpDir, 'feature')
                    
                    os.mkdir(path, mode)
                    featureDir = str(aeExpDir/'feature')
                    print('"Feature" directory is created')
                    
                    
                    #check threshold file exist or not
                    if (Path(thresholdFeatureFile).exists()):
                        #--- Append values
                        with open(thresholdFeatureFile, 'a') as csv_file:
                            dict_object = csv.writer(csv_file, delimiter=',')
                            dict_object.writerow(thresholdArrToSave)
                            csv_file.close()
                    else:
                        np.savetxt(thresholdFeatureFile,thresholValuWithFileName, delimiter=",", fmt='%s')

directoryPath = os.getcwd()+'/data/polymer-ae/denoised/'
readAndProcessDataForThresolding(directoryPath)
    

