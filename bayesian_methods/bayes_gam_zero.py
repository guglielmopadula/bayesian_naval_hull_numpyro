import os
import multiprocessing

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA
import dill
import jax
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
import scipy
import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from jax.lib import xla_bridge
import os
import multiprocessing
from numpyro.infer import log_likelihood


NUM_BAYES_SAMPLES=3000
NUM_WARMUP=1000#27000
NUM_CHAINS=4

classic_coeffs=np.load("npy_files/gam_coef.npy")
alpha_mean=classic_coeffs[-1]
beta_mean=classic_coeffs[:-1]
alpha_mean=jnp.array(alpha_mean.reshape(1,1))
beta_mean=jnp.array(beta_mean.reshape(-1,1))

def invert_permutation(p):
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def run_inference(model, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model,step_size=0.0001)
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_BAYES_SAMPLES,
        num_chains=NUM_CHAINS,
        progress_bar=True,
    )
    mcmc.run(rng_key, X, Y)
    divergences=jnp.sum(mcmc.get_extra_fields()["diverging"])
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples(group_by_chain=True),divergences
'''
# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]
'''


# create artificial regression dataset


def eta_constructor(m,d):
    if d//2==0:
        def simple_eta(x):
            return (-1)**(m+1+d//2)/(2**(2*m-1)*np.pi**(d//2)*np.math.factorial(m-1)*np.math.factorial(m+d//2-1))*x**(2*m-d)*np.log(x)
    
    if d//2!=0:
        def simple_eta(x):
            return scipy.special.gamma(d/2-m)/(2**(2*m)*np.pi**(d//2)*np.math.factorial(m-1))*x**(2*m-d)
        
    return simple_eta   


def model(X,Y):
    N,D_X=X.shape
    alpha = numpyro.sample("alpha", dist.Normal(0, 0.1*np.ones((1, 1))))
    beta = numpyro.sample("beta", dist.Normal(0,0.1*jnp.ones((D_X, 1))))
    logit=jnp.matmul(X,beta)+alpha
    y=numpyro.sample("Y",dist.BernoulliLogits(logit).to_event(1),obs=Y)
    return y

def compute_features(train):
    NUM_DATA=train.shape[0]
    NUM_FEATURES=train.shape[1]

    m=4 #must be 2*m>d+1
    simple_eta=eta_constructor(m,NUM_FEATURES)
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
    alpha_coeff=T
    X_train=np.concatenate((delta_coeff,alpha_coeff),axis=1)
    return X_train


def my_log_likehood(posterior_samples,train,target_train):
    train_features=compute_features(train)
    return log_likelihood(model,posterior_samples,train_features,target_train)["Y"]


def partition0(max_range, S):
    K = len(max_range)
    return np.array([i for i in itertools.product(*(range(i+1) for i in max_range)) if sum(i)<=S])            
if __name__ == "__main__":

    print(xla_bridge.get_backend().platform)
    print(jax.device_count())
    print(jax.local_device_count())
    numpyro.set_host_device_count(4)

    m=4 #must be 2*m>d+1
    data=np.load("npy_files/data.npy",allow_pickle=True).item()
    target_train=data["target_train"]
    train=data["train"]
    test=data["test"]
    target_test=data["target_test"]
    NUM_DATA=train.shape[0]
    NUM_FEATURES=train.shape[1]
    X_train=compute_features(train)
    X_test=compute_features(test)

    classic_coeffs=np.load("npy_files/gam_coef.npy")
    alpha_mean=classic_coeffs[-1]
    beta_mean=classic_coeffs[:-1]
    alpha_mean=jnp.array(alpha_mean.reshape(1,1))
    beta_mean=jnp.array(beta_mean.reshape(-1,1))


    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    shape=X_train[0].shape[0]
    posterior_samples,divergences=run_inference(model,rng_key,X_train,target_train)
    alpha=posterior_samples["alpha"].reshape(NUM_CHAINS,NUM_BAYES_SAMPLES,-1)
    beta=posterior_samples["beta"].reshape(NUM_CHAINS,NUM_BAYES_SAMPLES,-1)


    predictive = Predictive(model, posterior_samples, return_sites=["Y"])
    y_fitted_dist=predictive(random.PRNGKey(0), X_train,None)["Y"]
    y_fitted_dist=y_fitted_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_fitted_prob=np.mean(y_fitted_dist,axis=0)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_dist=predictive(random.PRNGKey(0), X_test,None)["Y"]
    y_predictive_dist=y_predictive_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_predictive_prob=np.mean(y_predictive_dist,axis=0)
    y_predictive=np.round(y_predictive_prob)
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))
    np.save("./npy_files/bayes_gam_zero.npy",{"posterior_samples":posterior_samples,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":NUM_CHAINS,"num_bayes_samples":NUM_BAYES_SAMPLES,"divergences":divergences})
