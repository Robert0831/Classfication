import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset50Loader
import cv2
from skimage.feature import hog

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
#validation_loader=Dataset50Loader('val.txt')

fnn=FNN()
fnn.load_state_dict(torch.load("./nnmodel_100.pth"))
optimize=torch.optim.SGD(fnn.parameters(),lr=0.01)
lossfc=nn.CrossEntropyLoss()
#hog=cv2.HOGDescriptor((32,32),(16,16),(2,2),(4,4),1)
for epoch in range(5):
    for batch_idx, (img,data) in enumerate(train_loader):
        fnn.train()
        imgs=[]
        for i in range(img.size(0)):
            aa=np.array(img[i],dtype=np.uint8)
            temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
            #256*256->216
            imgs.append(temp)

        out=fnn(torch.Tensor(np.array(imgs)))
        loss=lossfc(out,data)
        optimize.zero_grad
        loss.backward()
        optimize.step()
        if batch_idx%50==0:
            print(f'epoch={epoch} ,batch={batch_idx},loss={loss}')
        if batch_idx==0:
            fnn.eval()
            with torch.no_grad():
                out=out.numpy()
                out=np.argmax(out,1)
                print(out)
                print(data)
    torch.save(fnn.state_dict(), "nnmodel_%d.pth" %(epoch))



# clf = load('trainmodel.joblib') 
# y_pred = clf.predict(imgs)
# print(y_pred)
# print(datas)