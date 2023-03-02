import torch
import numpy as np
from dataset import Dataset50Loader
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import cv2
from joblib import dump, load
from skimage.feature import hog

train_loader=Dataset50Loader('train2.txt')
#validation_loader=Dataset50Loader('val.txt')

clf = svm.SVC(kernel='linear', C=1, probability=True)
#clf = load('trainmodel.joblib')
#clf=SGDClassifier(loss='log', penalty="l2")
imgs=[]
datas=[]
for batch_idx, (img,data) in enumerate(train_loader):
    print(batch_idx)
    for i in range(img.size(0)):
        aa=np.array(img[i],dtype=np.uint8)
        temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
        imgs.append(temp)
        datas.append(int(data[i]))
    #clf.partial_fit(imgs,datas,classes=np.unique(datas))
clf.fit(imgs, datas)
dump(clf, 'trainmodel.joblib') 

# clf = load('trainmodel.joblib') 
# y_pred = clf.predict(imgs)
# print(y_pred)
# print(datas)