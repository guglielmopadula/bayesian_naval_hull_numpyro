from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

data=np.load("./npy_files/data.npy",allow_pickle=True).item()
target_train=data["target_train"]
train=data["train"]
test=data["test"]
target_test=data["target_test"]

clf = LogisticRegression(random_state=0,penalty="l2",max_iter=500).fit(train, target_train)
print(clf.coef_)
print(clf.intercept_)
np.save("./npy_files/classiclogisticregression_coef.npy",np.concatenate((clf.coef_.reshape(-1),clf.intercept_.reshape(-1))))
print(confusion_matrix(target_train,clf.predict(train)))
print(confusion_matrix(target_test,clf.predict(test)))