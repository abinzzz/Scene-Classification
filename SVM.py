import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def train(data,label):
    data=np.array(data,dtype=np.float32)
    label=np.array(label,dtype=np.int32)
    svc=svm.SVC(kernel='rbf',C=1000,decision_function_shape='ovo')
    svc.fit(data,label)
    return svc

def predict(data,svm):
    data = np.array(data, dtype='float32')
    prediction = svm.predict(data)

    return prediction

def evaluate(prediction, label):
    accuracy = accuracy_score(label, prediction)
    confuse_matrix = confusion_matrix(label, prediction)
    report = classification_report(label, prediction)

    return accuracy, confuse_matrix,report