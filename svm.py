import os
from sklearn.svm import LinearSVC
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SVM:

    def __init__(self):
        self.model = LinearSVC()

    def train(self):
        img_list = np.array([])
        target_list = np.array([])
                
        sub_folders = [name for name in os.listdir('./image_data') if os.path.isdir(os.path.join('./image_data', name))]
        self.labels = sub_folders

        fileCount = 0
        imgsize = 0

        for i in range(len(sub_folders)):
            folder = sub_folders[i]
            files = os.listdir(f'./image_data/{folder}/')

            for j in range(len(files)):                
                img = cv2.imread(f'image_data/{folder}/{files[j]}')[:, :, 0]

                if i == 0 and j == 0:
                    imgsize = img.shape[0] * img.shape[1]
                img = img.reshape(-1)
                img_list = np.append(img_list, [img])
                target_list = np.append(target_list, sub_folders.index(folder))
                fileCount += 1

        img_list = img_list.reshape(fileCount, imgsize)
        # self.model.fit(img_list, target_list)
        # print("Model successfully trained!")

        xtrain, xtest, ytrain, ytest=train_test_split(img_list, target_list, test_size=0.15)

        print(self.model)

        self.model.fit(xtrain, ytrain)
        score = self.model.score(xtrain, ytrain)
        print("Score: ", score)

        cv_scores = cross_val_score(self.model, xtrain, ytrain, cv=10)
        print("CV average score: %.2f" % cv_scores.mean())

        ypred = self.model.predict(xtest)

        cm = confusion_matrix(ytest, ypred)
        print(cm)

        cr = classification_report(ytest, ypred)
        print(cr) 

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = self.labels)

        cm_display.plot()
        plt.title("svm confusion matrix")
        plt.show()

    def predict(self, frame):
        
        img = frame.reshape(-1)
        prediction = self.model.predict([img])
        
        result = self.labels[int(prediction[0])]

        return result