
import numpy as np
from dataset import Dataset50Loader
import lightgbm as lgb
from skimage.feature import hog
from joblib import dump, load



train_loader=Dataset50Loader('train2.txt')
imgs=[]
datas=[]
for batch_idx, (img,data) in enumerate(train_loader):
    print(batch_idx)
    for i in range(img.size(0)):
        aa=np.array(img[i],dtype=np.uint8)
        temp=hog(aa,orientations=6,pixels_per_cell=(64,64),cells_per_block=(2,2))
        imgs.append(np.array(temp))
        datas.append(int(data[i]))

traindata=lgb.Dataset(np.array(imgs), label=datas)
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=10
params['num_class']=50 #no.of unique values in the target class not inclusive of the end value

#set epoh 
clf=lgb.train(params,traindata,100)
clf.save_model('lgbmodel.txt')



###test
# clf=lgb.Booster(model_file='lgbmodel.txt')
# yy=clf.predict(imgs)
# y_pred_1 = [np.argmax(line) for line in yy]
# print(y_pred_1)
# print(datas)