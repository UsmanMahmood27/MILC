'''
Aug_Samples_Phase_2_Ratio_2_VAR_SVAR_Samples implies slicing each time series into
five slices, each slice is in effect considered as a subject.
'''

import numpy as np
import pandas as pd
import os

def load_Augmented_Data():
    if not os.path.exists('../svar_2_datafileaugmented_tenWS.npz'):
        data = np.zeros((2000, 10, 4000))
        finalData = np.zeros((2000, 400, 10, 10))

        for p in range(400):
            filename = '../Aug_Samples_Phase_2_Ratio_2_VAR_SVAR_Samples/TSDataCSV' + str(p) + '.csv'
            print(filename)
            df = pd.read_csv(filename)
            df = df.to_numpy()
            data[p*5:p*5+5, :, :] = df.reshape(5, 10, 4000)  #Five slices per TS save them as 5 TS samples of shorter length

        for i in range(400):
            for j in range(400):
                finalData[i*5:i*5+5, j, :, :] = data[i*5:i*5+5, :, j * 10:j * 10 + 10]

        print(finalData.shape)
        with open('../svar_2_datafileaugmented_tenWS.npz', 'wb') as file:
            np.save(file, finalData)
            print('Augmented data loaded and saved successfully...')
    else:
        with open('../svar_2_datafileaugmented_tenWS.npz', 'rb') as file:
            finalData = np.load(file)
            print('Augmented data loaded successfully...')
    return finalData

