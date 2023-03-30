import cv2
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from scipy.io import loadmat
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


mnist_path = './mnist_data/mnist-original.mat'
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
print("mnist data loaded!")

X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

logReg_clf = LogisticRegression(max_iter = 1000)
logReg_clf.fit(X_train, y_train) # train the classifier

score = logReg_clf.score(X_train, y_train)
print("Accuracy Score: ", score)

y_train_pred = logReg_clf.predict(X_test)  
#y_train_pred = cross_val_predict(logReg_clf, X_test, y_test, cv=3)  
cm = confusion_matrix(y_test, y_train_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title("LogisticRegression confusion matrix")
plt.show()

while 1 == 1:
    if cv2.waitKey(5) & 0xFF == 27:
        break;


