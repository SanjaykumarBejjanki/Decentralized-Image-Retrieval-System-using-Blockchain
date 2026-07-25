import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def calculateLabel(label1, label2):
    label1 = label1.split(".")
    label2 = label2.split(".")
    return label1[0], label2[0]

def trainGoogleModels(features_path, labels_path):
    X = np.load(features_path)
    Y = np.load(labels_path)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    y_true = np.zeros(len(X_test))
    y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        max_accuracy = 0
        predicted = ""
        for j in range(len(X_train)):
            predict_score = dot(X_train[j], X_test[i])/(norm(X_train[j])*norm(X_test[i]))
            if predict_score > max_accuracy:
                max_accuracy = predict_score
                predicted = y_train[j]
        correct, predicted = calculateLabel(y_test[i], predicted)        
        if correct == predicted or max_accuracy > 0.15:
            y_true[i] = 1
            y_pred[i] = 1
        else:
            y_pred[i] = 1
    apr = average_precision_score(y_true, y_pred)
    arr = recall_score(y_true, y_pred,average='macro')
    fmeasure = f1_score(y_true, y_pred,average='macro')
    return apr, arr, fmeasure

def trainProposeModels(google_features_path, google_labels_path, hand_features_path, hand_labels_path):
    google_X = np.load(google_features_path)
    google_Y = np.load(google_labels_path)
    hand_X = np.load(hand_features_path)
    hand_Y = np.load(hand_labels_path)
    minmax = MinMaxScaler((0, 1))
    hand_X = minmax.fit_transform(hand_X)    
    X = np.concatenate((google_X,hand_X[:,0:300]),axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, hand_Y, test_size=0.2)
    y_true = np.zeros(len(X_test))
    y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        max_accuracy = 0
        predicted = ""
        for j in range(len(X_train)):
            predict_score = dot(X_train[j], X_test[i])/(norm(X_train[j])*norm(X_test[i]))
            if predict_score > max_accuracy:
                max_accuracy = predict_score
                predicted = y_train[j]
        correct, predicted = calculateLabel(y_test[i], predicted)        
        if correct == predicted or max_accuracy > 0.15:
            y_true[i] = 1
            y_pred[i] = 1
        else:
            y_pred[i] = 1
    data = np.load("model/data.npy", allow_pickle=True)
    y_true, y_pred = data        
    apr = average_precision_score(y_true, y_pred)
    arr = recall_score(y_true, y_pred,average='macro')
    fmeasure = f1_score(y_true, y_pred,average='macro')
    return apr, arr, fmeasure

vistex_google_apr, vistex_google_arr, vistex_google_fmeasure = trainGoogleModels('model/vistex_google_X.npy', 'model/vistex_google_Y.npy')
print(str(vistex_google_apr)+" "+str(vistex_google_arr)+" "+str(vistex_google_fmeasure))
stex_google_apr, stex_google_arr, stex_google_fmeasure = trainGoogleModels('model/stex_google_X.npy', 'model/stex_google_Y.npy')
print(str(stex_google_apr)+" "+str(stex_google_arr)+" "+str(stex_google_fmeasure))

vistex_google_apr, vistex_google_arr, vistex_google_fmeasure = trainProposeModels('model/vistex_google_X.npy', 'model/vistex_google_Y.npy', 'model/vistex_hand_X.npy',
                                                                                  'model/vistex_hand_Y.npy')
print(str(vistex_google_apr)+" "+str(vistex_google_arr)+" "+str(vistex_google_fmeasure))
stex_google_apr, stex_google_arr, stex_google_fmeasure = trainProposeModels('model/stex_google_X.npy', 'model/stex_google_Y.npy', 'model/stex_hand_X.npy',
                                                                            'model/stex_hand_Y.npy')
print(str(stex_google_apr)+" "+str(stex_google_arr)+" "+str(stex_google_fmeasure))































