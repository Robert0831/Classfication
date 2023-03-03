import torch
import numpy as np
import torch.nn as nn
from dataset import Dataset50Loader
from sklearn.model_selection import train_test_split
from sklearn import svm
import cv2
from joblib import dump, load
from skimage.feature import hog
import lightgbm as lgb

val_loader=Dataset50Loader('val.txt')
test_loader=Dataset50Loader('test.txt')
total=len(val_loader.dataset) #nums data
#total=len(val_loader) nums batch
def most_find(sequence, n):
    lst = sorted(range(len(sequence)), key=lambda x:sequence[x], reverse=True)
    return lst[:n]

######################## SVM ########################
clf = svm.SVC(kernel='linear', C=1, probability=True)
clf = load('trainmodel.joblib')
svmans_1=0
svmans_5=0
for batch_idx, (img,data) in enumerate(test_loader):
    imgs=[]
    datas=[]
    for i in range(img.size(0)):
        aa=np.array(img[i],dtype=np.uint8)
        temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
        imgs.append(np.array(temp))
        datas.append(int(data[i]))
    y_pred_1 = clf.predict(imgs)
    y_pred_5=clf.predict_proba(imgs)
    for j in range(img.size(0)):
        if y_pred_1[j]==datas[j]:
            svmans_1+=1
        if datas[j] in most_find(y_pred_5[j],5):
            svmans_5+=1           

#print(f'svm top-1:{svmans_1} rate:{round(svmans_1/total,2)},top-5:{svmans_5} rate:{round(svmans_5/total,2)} in validation') # top1:51 top5:154
print(f'svm top-1:{svmans_1} rate:{round(svmans_1/total,2)},top-5:{svmans_5} rate:{round(svmans_5/total,2)} in test') # top1:53 top5:168
######################################################



###################### NN MODEL #######################
class FNN(nn.Module):
    def __init__(self) :
        super(FNN,self).__init__()
        self.allnn=nn.Sequential(
                nn.Linear(216,216),
                nn.ReLU(),
                nn.Linear(216,100),
                nn.ReLU(),
                nn.Linear(100,50),
                nn.Softmax(dim=1),
        )
    def forward(self,x):
        output=self.allnn(x)
        return output
fnn=FNN()
fnn.load_state_dict(torch.load("./nnmodel_test.pth"))

fnn.eval()
nnans_1=0
nnans_5=0
for batch_idx, (img,data) in enumerate(test_loader):
    imgs=[]
    datas=[]
    for i in range(img.size(0)):
        aa=np.array(img[i],dtype=np.uint8)
        temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
        #256*256->216
        imgs.append(np.array(temp))
        datas.append(int(data[i]))
    with torch.no_grad():
        out=fnn(torch.Tensor(np.array(imgs)))
        out=out.numpy()
        out_1=np.argmax(out,1)
    for j in range(img.size(0)):
        if out_1[j]==datas[j]:
            nnans_1+=1
        if datas[j] in most_find(out[j],5):
            nnans_5+=1
#print(f'fnn top-1:{nnans_1} rate:{round(nnans_1/total,2)},top-5:{nnans_5} rate:{round(nnans_5/total,2)} in validation') # top1:9 top5:45
print(f'fnn top-1:{nnans_1} rate:{round(nnans_1/total,2)},top-5:{nnans_5} rate:{round(nnans_5/total,2)} in test') # top1:9 top5:45
#######################################################


###################### LGB #######################
lgbmodle=lgb.Booster(model_file='lgbmodel.txt')
lgbans_1=0
lgbans_5=0
for batch_idx, (img,data) in enumerate(test_loader):
    imgs=[]
    datas=[]
    for i in range(img.size(0)):
        aa=np.array(img[i],dtype=np.uint8)
        temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
        imgs.append(np.array(temp))
        datas.append(int(data[i]))
    yy=lgbmodle.predict(imgs)
    y_pred_1 = [np.argmax(line) for line in yy]
    for j in range(img.size(0)):
        if y_pred_1[j]==datas[j]:
            lgbans_1+=1
        if datas[j] in most_find(yy[j], 5):
            lgbans_5+=1
#print(f'lgb top-1:{lgbans_1} rate:{round(lgbans_1/total,2)},top-5:{lgbans_5} rate:{round(lgbans_5/total,2)} in validation') # top1:53 top5:162
print(f'lgb top-1:{lgbans_1} rate:{round(lgbans_1/total,2)},top-5:{lgbans_5} rate:{round(lgbans_5/total,2)} in test') # top1:67 top5:152

#################################################






