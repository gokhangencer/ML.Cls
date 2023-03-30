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
from sklearn.neighbors import KNeighborsClassifier


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

print("X", X)
print("Y", y)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

knn_clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn_clf.fit(X_train, y_train)  # train the classifier

# cok uzun suruyor ve cpu %80 lere vurdu
# score = knn_clf.score(X_train, y_train)
# print("Accuracy Score: ", score)

y_train_pred = knn_clf.predict(X_test)
# y_train_pred = cross_val_predict(logReg_clf, X_test, y_test, cv=3)
cm = confusion_matrix(y_test, y_train_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
plt.title("KNeighborsClassifier confusion matrix")
plt.show()

while 1 == 1:
    if cv2.waitKey(5) & 0xFF == 27:
        break
