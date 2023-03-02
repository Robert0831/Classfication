# Image Classfication

Train model,use [train2.txt](https://github.com/Robert0831/Classfication/blob/main/train2.txt) data ,it is [train.txt](https://github.com/Robert0831/Classfication/blob/main/train.txt) being shuffled

To train deeplearning  model ,directly execution [nn.py](https://github.com/Robert0831/Classfication/blob/main/nn.py)

To train svm model ,directly execution [train.py](https://github.com/Robert0831/Classfication/blob/main/train.py)

To train Lightgbm model,directly execution [trainlgb.py](https://github.com/Robert0831/Classfication/blob/main/trainlgb.py)

----------------------------------------------------------------------------
Testing

To test three model directly execution [test.py](https://github.com/Robert0831/Classfication/blob/main/test.py)

To test on validation set or test set you need to change xxx to "val_loader" or "test_loader"

->  for batch_idx, (img,data) in enumerate(xxx):

----------------------------------------------------------------------------

Pretrain model download

Lightgbm:[lgbmodel.txt](https://github.com/Robert0831/Classfication/blob/main/lgbmodel.txt)

SVM:https://drive.google.com/file/d/1D99LLhI_kxqGylxLPyOlHawjdxv4r6fv/view?usp=share_link

DEEP LEARNING:https://drive.google.com/file/d/1bf072phndgx-e40jNyDV4l2czKhXZuUa/view?usp=share_link
