
import numpy as np
import os
import pandas as pd
from pathlib import Path 
import glob


def calculatethreshold(denoisedDataArr):
    return np.mean(denoisedDataArr) + 2.5 * np.std(denoisedDataArr)


 
def SaveThresold(directoryPath):
    threshold_data = []
    for root, dirs, files in os.walk(directoryPath):
        for denoisedDirName in dirs:
            txtFiles = glob.glob(root+denoisedDirName+'/*.txt')
            for file in txtFiles:
                denoisedDataDf = np.loadtxt(file)
                thresholdvalues= calculatethreshold(denoisedDataDf)
                fileName =  file.replace(root+denoisedDirName+'/','')
                threshold_data.append((fileName, thresholdvalues))
                threshold_filepath = os.path.join(root, denoisedDirName, f'{fileName}_threshold.txt')
                np.savetxt(threshold_filepath, [thresholdvalues], delimiter=',')

                
                    


    threshold_file = os.path.join(directoryPath, 'threshold_values.csv')
    df= pd.DataFrame(threshold_data, columns=['File Name', 'Threshold Value'])
    df.to_csv(threshold_file, index=False, header=True)
    return "Threshold values saved."

directoryPath = os.getcwd()+'/data/polymer-ae'
SaveThresold(directoryPath)


 

             







                

"""
                if ((aeExpDir/'threshholdvalues').exists()):
                    denoisedDir = str(aeExpDir/'thresholdvalues')
                    
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
                        thresholdvaluesFile = denoisedDir + "/" + rowDirName + "/" + fileName
                        #denoisedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(thresholdvaluesFile, thresholdArr, delimiter=",")

                else:
                    mode = 0o777
  
                    # Path
                    path = os.path.join(aeExpDir, 'threshold')
                    
                    os.mkdir(path, mode)
                    denoisedDir = str(aeExpDir/'threshold')
                    print('"threshold" directory is created')
                    
                    #check
                    if (Path(aeExpDir/'threshold'/rowDirName).exists()):
                        #denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName + '.csv'
                        denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName
                        #denoisedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(thresholdvaluesFile, thresholdArr, delimiter=",")
                    else:
                        mode = 0o777
  
                        # Path
                        path = os.path.join(thresholdDir, denoisedDirName)
                        os.mkdir(path, mode)        
                        print(rowDirName + ' Directory is created')
                    
                        #Save the denoised data to a new CSV file
                        #denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName + '.csv'
                        denoisedFile = denoisedDir + "/" + rowDirName + "/" + fileName
                        #denoisedDataDf.to_csv(denoisedFile, index=False)
                        np.savetxt(denoisedFile, denoisedDataArr, delimiter=",")
"""      