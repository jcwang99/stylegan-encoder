import os
import pickle
import config
import dnnlib
import gzip
import json

# 线性模型库
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# 执行线性回归,得到性别向量 gender_direction 
clf = LogisticRegression(class_weight='balanced').fit(X_data, y_gender_data)
gender_direction = clf.coef_.reshape((18, 512))
import numpy as np
from tqdm import tqdm_notebook
import warnings
import matplotlib.pylab as plt
%matplotlib inline
warnings.filterwarnings("ignore")

# 加载数据集
model_dir='./models/latent_training_data.pkl'
qlatent_data, dlatent_data, labels_data = pickle.load(open(model_dir,"rb"))
# qlatent_data Z (20307, 512)
# dlatent_data W+ (20307, 18, 512)
# labels_data 20307

# 格式化数据
X_data = dlatent_data.reshape((-1, 18*512))
y_age_data = np.array([x['faceAttributes']['age'] for x in labels_data])
y_gender_data = np.array([x['faceAttributes']['gender'] == 'male' for x in labels_data])
assert(len(X_data) == len(y_age_data) == len(y_gender_data))
len(X_data)

# 线性模型库
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# 执行线性回归,得到性别向量 gender_direction 
clf = LogisticRegression(class_weight='balanced').fit(X_data, y_gender_data)
gender_direction = clf.coef_.reshape((18, 512))
