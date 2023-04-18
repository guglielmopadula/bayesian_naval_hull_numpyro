#Using default thing plate splines
#Thin plate splines regression Wood
import scipy
import numpy as np
import itertools
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

data=np.load("./npy_files/data.npy",allow_pickle=True).item()

target_train=data["target_train"]
train=data["train"]
test=data["test"]
target_test=data["target_test"]
NUM_DATA=train.shape[0]
NUM_FEATURES=train.shape[1]
print(NUM_FEATURES)

def eta_constructor(m,d):
    if d//2==0:
        def simple_eta(x):
            return (-1)**(m+1+d//2)/(2**(2*m-1)*np.pi**(d//2)*np.math.factorial(m-1)*np.math.factorial(m+d//2-1))*x**(2*m-d)*np.log(x)
    
    if d//2!=0:
        def simple_eta(x):
            return scipy.special.gamma(d/2-m)/(2**(2*m)*np.pi**(d//2)*np.math.factorial(m-1))*x**(2*m-d)
        
    return simple_eta   

m=4
simple_eta=eta_constructor(m,NUM_FEATURES)







def partition0(max_range, S):
    K = len(max_range)
    return np.array([i for i in itertools.product(*(range(i+1) for i in max_range)) if sum(i)<=S])            
part=partition0([m,m,m,m,m],m-1)
part=part.T
part=part.reshape(1,part.shape[0],part.shape[1])
part=part.repeat(NUM_DATA,axis=0)
M=part.shape[2]
k=M+1



Distance_train=pairwise_distances(train)
E_train=simple_eta(Distance_train)
tmp=train.reshape(NUM_DATA,NUM_FEATURES,-1).repeat(M,axis=2)
T=np.prod(tmp**part,axis=1)
w,v=np.linalg.eig(E_train)
u=v.T
uk=u[:,:k]
dk=np.diag(w[:k])
Zk=scipy.linalg.null_space(T.T@uk)
delta_coeff=uk@dk@Zk
print(delta_coeff.shape)
alpha_coeff=T
X_train=np.concatenate((delta_coeff,alpha_coeff),axis=1)



Distance_test=pairwise_distances(test)
E_test=simple_eta(Distance_test)
tmp=test.reshape(NUM_DATA,NUM_FEATURES,-1).repeat(M,axis=2)
T=np.prod(tmp**part,axis=1)
w,v=np.linalg.eig(E_test)
u=v.T
uk=u[:,:k]
dk=np.diag(w[:k])
Zk=scipy.linalg.null_space(T.T@uk)
delta_coeff=uk@dk@Zk
alpha_coeff=T
print(delta_coeff.shape)
X_test=np.concatenate((delta_coeff,alpha_coeff),axis=1)

clf = LogisticRegression(random_state=0,penalty="l2",max_iter=500).fit(X_train, target_train)
print(confusion_matrix(target_train,clf.predict(X_train)))
print(confusion_matrix(target_test,clf.predict(X_test)))

np.save("./npy_files/gam_coef.npy",np.concatenate((clf.coef_.reshape(-1),clf.intercept_.reshape(-1))))
