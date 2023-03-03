import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset50Loader
import cv2
from skimage.feature import hog
import pandas as pd

def most_find(sequence, n):
    lst = sorted(range(len(sequence)), key=lambda x:sequence[x], reverse=True)
    return lst[:n]

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

train_loader=Dataset50Loader('train2.txt')
val_loader=Dataset50Loader('val.txt')
totalt=len(train_loader.dataset) #nums data
totalv=len(val_loader.dataset)

fnn=FNN()
#fnn.load_state_dict(torch.load("./nnmodel_test.pth"))
optimize=torch.optim.SGD(fnn.parameters(),lr=0.001)
lossfc=nn.CrossEntropyLoss()
#hog=cv2.HOGDescriptor((32,32),(16,16),(2,2),(4,4),1)
for epoch in range(10):
    nnans_1t=0
    nnans_5t=0
    nnans_1v=0
    nnans_5v=0
    for batch_idx, (img,data) in enumerate(train_loader):
        fnn.train()
        imgs=[]
        datas=[]
        for i in range(img.size(0)):
            aa=np.array(img[i],dtype=np.uint8)
            temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
            #256*256->216
            imgs.append(temp)
            datas.append(int(data[i]))
        out=fnn(torch.Tensor(np.array(imgs)))
        loss=lossfc(out,data)
        optimize.zero_grad
        loss.backward()
        optimize.step()

        with torch.no_grad():
            out=fnn(torch.Tensor(np.array(imgs)))
            out=out.numpy()
            out_1=np.argmax(out,1)

        for j in range(img.size(0)):
            if out_1[j]==datas[j]:
                nnans_1t+=1
            if datas[j] in most_find(out[j],5):
                nnans_5t+=1


    for batch_idxv, (imgv,datav) in enumerate(val_loader):
        imgsv=[]
        datasv=[]

        for i in range(imgv.size(0)):
            aa=np.array(imgv[i],dtype=np.uint8)
            temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
            #256*256->216
            imgsv.append(np.array(temp))
            datasv.append(int(datav[i]))
        with torch.no_grad():
            out=fnn(torch.Tensor(np.array(imgsv)))
            out=out.numpy()
            out_1=np.argmax(out,1)
        for j in range(imgv.size(0)):
            if out_1[j]==datasv[j]:
                nnans_1v+=1
            if datasv[j] in most_find(out[j],5):
                nnans_5v+=1

    print(f'Epoch:{epoch} ; Training top-1 accuracy:{round(nnans_1t/totalt,5)},top-5 accuracy:{round(nnans_5t/totalt,5)} ; Validation top-1 accuracy:{round(nnans_1v/totalv,5)},top-5 accuracy:{round(nnans_5v/totalv,5)}') 
    # logger = open('score.txt', 'a')
    # logger.write('%f %f %f %f\n'%(round(nnans_1t/totalt,5),round(nnans_5t/totalt,5),round(nnans_1v/totalv,5),round(nnans_5v/totalv,5)))
    # logger.close()

        # if batch_idx%50==0:
        #     print(f'epoch={epoch} ,batch={batch_idx},loss={loss}')
        # if batch_idx==0:
        #     fnn.eval()
        #     with torch.no_grad():
        #         out=out.numpy()
        #         out=np.argmax(out,1)
        #         print(out)
        #         print(data)

    torch.save(fnn.state_dict(), "nnmodel_test.pth" )



