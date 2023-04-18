from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA
from sklearn.metrics import pairwise_distances
import dill
import jax
from sklearn.metrics import confusion_matrix
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from jax.lib import xla_bridge
NUM_BAYES_SAMPLES=1000
NUM_WARMUP=3000
NUM_CHAINS=4
from numpyro.infer import log_likelihood

def invert_permutation(p):
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def run_inference(model, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model,step_size=0.0001) #0.0001
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
classic_coeffs=np.load("npy_files/nn_params.npy",allow_pickle=True).item()
bias0_prior=classic_coeffs['dense_0_bias']
matrix0_prior=classic_coeffs['dense_0_matrix']
bias1_prior=classic_coeffs['dense_1_bias']
matrix1_prior=classic_coeffs['dense_1_matrix']
bias2_prior=classic_coeffs['dense_2_bias']
matrix2_prior=classic_coeffs['dense_2_matrix']


def model(X,Y):
    N,D_X=X.shape
    bias0 = numpyro.sample("bias0", dist.Normal(bias0_prior, 0.001*np.ones((1, bias0_prior.shape[0]))))#0.001
    bias1 = numpyro.sample("bias1", dist.Normal(bias1_prior, 0.001*np.ones((1, bias1_prior.shape[0]))))#0.001
    bias2 = numpyro.sample("bias2", dist.Normal(bias2_prior, 0.001*np.ones((1, bias2_prior.shape[0]))))#0.001
    matrix0 = numpyro.sample("matrix0", dist.Normal(matrix0_prior,0.001*jnp.ones((matrix0_prior.shape[0], matrix0_prior.shape[1]))))
    matrix1 = numpyro.sample("matrix1", dist.Normal(matrix1_prior,0.001*jnp.ones((matrix1_prior.shape[0], matrix1_prior.shape[1]))))
    matrix2 = numpyro.sample("matrix2", dist.Normal(matrix2_prior,0.001*jnp.ones((matrix2_prior.shape[0], matrix2_prior.shape[1]))))
    hid1=jnp.matmul(X,matrix0)+bias0
    hid1=jnp.tanh(hid1)
    hid2=jnp.matmul(hid1,matrix1)+bias1
    hid2=jnp.tanh(hid2)
    hid3=jnp.matmul(hid2,matrix2)+bias2
    p=jnp.exp(hid3)[...,1]/jnp.sum(jnp.exp(hid3),axis=-1)
    y=numpyro.sample("Y",dist.Bernoulli(p).to_event(1),obs=Y)
    return y

def my_log_likehood(posterior_samples,X,Y):
    bias0 = posterior_samples["bias0"].reshape(-1,*posterior_samples["bias0"].shape[2:])
    bias1 = posterior_samples["bias1"].reshape(-1,*posterior_samples["bias1"].shape[2:])
    bias2 = posterior_samples["bias2"].reshape(-1,*posterior_samples["bias2"].shape[2:])
    matrix0 = posterior_samples["matrix0"].reshape(-1,*posterior_samples["matrix0"].shape[2:])
    matrix1 = posterior_samples["matrix1"].reshape(-1,*posterior_samples["matrix1"].shape[2:])
    matrix2 = posterior_samples["matrix2"].reshape(-1,*posterior_samples["matrix2"].shape[2:])
    hid1=jnp.matmul(X,matrix0)+bias0
    hid1=jnp.tanh(hid1)
    hid2=jnp.matmul(hid1,matrix1)+bias1
    hid2=jnp.tanh(hid2)
    hid3=jnp.matmul(hid2,matrix2)+bias2
    p=np.minimum(np.maximum(jnp.exp(hid3)[...,1]/jnp.sum(jnp.exp(hid3),axis=-1),0.000001),0.999999)
    return dist.Bernoulli(p).log_prob(Y)

if __name__ == "__main__":
    print(xla_bridge.get_backend().platform)
    print(jax.device_count())
    print(jax.local_device_count())
    numpyro.set_host_device_count(4)

    data=np.load("npy_files/data.npy",allow_pickle=True).item()
    target_train=data["target_train"]
    train=data["train"]
    test=data["test"]
    target_test=data["target_test"]








    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    shape=train[0].shape[0]
    posterior_samples,divergences=run_inference(model,rng_key,train,target_train)
    print(posterior_samples["matrix0"].shape)

    predictive = Predictive(model, posterior_samples, return_sites=["Y"])
    y_fitted_dist=predictive(random.PRNGKey(0), train,None)["Y"]
    y_fitted_dist=y_fitted_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_fitted_prob=np.mean(y_fitted_dist,axis=0)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_dist=predictive(random.PRNGKey(0), test,None)["Y"]
    y_predictive_dist=y_predictive_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_predictive_prob=np.mean(y_predictive_dist,axis=0)
    y_predictive=np.round(y_predictive_prob)
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))
    np.save("npy_files/bayesian_nn_mle.npy",{"posterior_samples":posterior_samples,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":NUM_CHAINS,"num_bayes_samples":NUM_BAYES_SAMPLES,"num_warmup":NUM_WARMUP,"divergences":divergences})
