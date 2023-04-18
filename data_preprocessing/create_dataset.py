import numpy as np
true=np.load("npy_files/positive_data_red.npy")
false=np.load("npy_files/negative_data_red.npy")
print(true.shape)
np.random.seed(2)
indextrue=np.random.choice(len(true), len(true)//2, replace=False)
indexfalse=np.random.choice(len(true), len(true)//2, replace=False)
mask_traintrue = np.zeros(true.shape[0], dtype=bool)
mask_traintrue[indextrue] = True
mask_testtrue = np.ones(true.shape[0], dtype=bool)
mask_testtrue[indextrue] = False
train_true = true[mask_traintrue]
test_true = true[mask_testtrue]
mask_trainfalse = np.zeros(false.shape[0], dtype=bool)
mask_trainfalse[indexfalse] = True
mask_testfalse = np.ones(false.shape[0], dtype=bool)
mask_testfalse[indextrue] = False
train_false = false[mask_trainfalse]
test_false = false[mask_testfalse]
train_perm=np.random.permutation(600)
test_perm=np.random.permutation(600)
train=np.concatenate((train_true,train_false))
test=np.concatenate((test_true,test_false))
target=np.concatenate((np.ones(300,dtype=np.int32),np.zeros(300,dtype=np.int32)))
target_train=target
train=train[train_perm]
target_train=target[train_perm]
test=test[test_perm]
target_test=target[test_perm]
data={"train":train,"test":test,"target_train":target_train,"target_test":target_test}
np.save("npy_files/data.npy",data)