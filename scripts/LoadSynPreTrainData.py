import numpy as np
import pandas as pd
import os

def loadData():
    if not os.path.exists('../pretrainingSynData.npz'):
        data = np.zeros((50, 10, 20000))
        finalData = np.zeros((50, 1000, 10, 20))

        for p in range(50):
            filename = '../TimeSeries/TSDataCSV' + str(p) + '.csv'
            print(filename)
            df = pd.read_csv(filename)
            data[p, :, :] = df

        for i in range(50):
            for j in range(1000):
                finalData[i, j, :, :] = data[i, :, j * 20:j * 20 + 20]

        print(finalData.shape)

        print(finalData.shape)
        with open('../pretrainingSynData.npz', 'wb') as file:
            np.save(file, finalData)
    else:
        with open('../pretrainingSynData.npz', 'rb') as file:
            finalData = np.load(file)
            print('Data loaded successfully...')
    return finalData


