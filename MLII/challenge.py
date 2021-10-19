import sys
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np


if __name__ == "__main__":
    # fetch data
    x_train = pd.read_csv("AMF_train_X.csv")
    if len(sys.argv) == 2:
        x_train = x_train.head(n=int(sys.argv[1]))

    y_train = pd.read_csv("AMF_train_Y.csv")
    x_test = pd.read_csv("AMF_test_X.csv")

    type_d = {}
    for index, row in y_train.iterrows():
        type_d[row['Trader']] = row['type']

    x_train2 = x_train.dropna()
    y_train2 = np.array([type_d[x] for x in x_train2['Trader']])
    x_train2 = x_train2.drop(columns=['Index', 'Share', 'Day', 'Trader'])

    print("X:", len(x_train2))
    print('Training started')

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(x_train2, y_train2)
    print('End of training')

    x_test2 = x_test.dropna()
    x_test2 = x_test2.drop(columns=['Index', 'Share', 'Day', 'Trader'])
    y_pred = svclassifier.predict(x_test2)

    x_test2 = x_test.dropna()
    names = x_test2['Trader'].unique()
    convention = {'HFT': 0, 'NON HFT': 1, 'MIX': 2}
    result = {e: [0, 0, 0] for e in names}
    i = 0
    for index, row in x_test2.iterrows():
        result[row['Trader']][convention[y_pred[i]]] += 1
        i += 1

    data = []
    for trader in names:
        n = sum(result[trader])
        if result[trader][0] > 0.85*n:
            data.append([trader, 'HFT'])
        elif result[trader][2] > 0.5*n:
            data.append([trader, 'MIX'])
        else:
            data.append([trader, 'NON HFT'])

    export_df = pd.DataFrame(data, columns=['Trader', 'type'])
    export_df.to_csv(r'result.csv', index=False)
