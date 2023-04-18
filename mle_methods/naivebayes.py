from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np



data=np.load("./npy_files/data.npy",allow_pickle=True).item()
target_train=data["target_train"]
train=data["train"]
test=data["test"]
target_test=data["target_test"]

gnb = GaussianNB().fit(train, target_train)
np.save("./npy_files/naivebayes_coef.npy",np.concatenate((gnb.theta_.reshape(2,-1),gnb.var_.reshape(2,-1))))
print(confusion_matrix(target_train,gnb.predict(train)))
print(confusion_matrix(target_test,gnb.predict(test)))