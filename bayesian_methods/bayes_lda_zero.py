from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro import handlers
import numpyro.distributions as dist
from numpyro import deterministic
from numpyro.infer import init_to_feasible
from numpyro.infer import MCMC, NUTS, SA
import dill
import jax
from sklearn.metrics import confusion_matrix
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax.lib import xla_bridge
NUM_BAYES_SAMPLES=1000#1000
NUM_WARMUP=3000#3000
NUM_CHAINS=4
NUM_DATA=600
D_X=5
from sklearn.preprocessing import StandardScaler

def invert_permutation(p):
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def run_inference(model, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model,step_size=0.001,init_strategy=init_to_feasible)
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


def model(X,Y):
    N=X.shape[0]
    D_X=X.shape[-1]
    mu=numpyro.sample("mu",dist.Normal(jnp.zeros((2,D_X)),jnp.ones((2,D_X))))
    covariance=numpyro.sample("covariance",dist.LKJCholesky(5,0.5))
    variance=numpyro.sample("variance",dist.Gamma(jnp.ones(D_X)))
    sigma=deterministic("sigma",jnp.outer(variance,variance)*covariance)
    x_hat=numpyro.sample("x_hat",dist.MultivariateNormal(mu[Y],sigma).to_event(1),obs=X) 
    return x_hat   

def predict(mu,sigma,X):
    X=X.reshape(1,NUM_DATA,-1)
    mu0=mu[:,0,:]
    mu1=mu[:,1,:]
    mu0=mu0.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,D_X)
    mu1=mu1.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,D_X)
    sigma=sigma.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,D_X,D_X)
    p_true=jnp.exp(dist.MultivariateNormal(mu1,sigma).log_prob(X))/(0.5*jnp.exp(dist.MultivariateNormal(mu0,sigma).log_prob(X))+0.5*jnp.exp(dist.MultivariateNormal(mu1,sigma).log_prob(X)))
    y_pred=dist.Bernoulli(p_true).sample(random.PRNGKey(0))
    return y_pred

def my_log_likehood(posterior_samples,X,Y):
    X=X.reshape(1,NUM_DATA,-1)
    mu=posterior_samples["mu"].reshape(-1,2,D_X)
    sigma=posterior_samples["sigma"].reshape(-1,1,D_X,D_X)
    mu0=mu[:,0,:]
    mu1=mu[:,1,:]
    mu0=mu0.reshape(-1,1,D_X)
    mu1=mu1.reshape(-1,1,D_X)
    sigma=sigma.reshape(-1,1,D_X,D_X)
    p_true=np.minimum(np.maximum(jnp.exp(dist.MultivariateNormal(mu1,sigma).log_prob(X))/(0.5*jnp.exp(dist.MultivariateNormal(mu0,sigma).log_prob(X))+0.5*jnp.exp(dist.MultivariateNormal(mu1,sigma).log_prob(X))),0.000001),0.999999)
    return dist.Bernoulli(p_true).log_prob(Y)

'''
# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]
'''

if __name__ == "__main__":

# create artificial regression dataset
    print(xla_bridge.get_backend().platform)
    print(jax.device_count())
    print(jax.local_device_count())
    numpyro.set_host_device_count(4)


    data=np.load("./npy_files/data.npy",allow_pickle=True).item()
    target_train=data["target_train"]
    train=data["train"]
    test=data["test"]
    target_test=data["target_test"]



    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    shape=train[0].shape[0]
    posterior_samples,divergences=run_inference(model,rng_key,train,target_train)

    mu=posterior_samples["mu"].reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,2,-1)
    sigma=posterior_samples["sigma"].reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,D_X,D_X)
    y_fitted_dist=predict(mu,sigma,train)
    y_fitted_prob=np.mean(y_fitted_dist,axis=0)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_dist=predict(mu,sigma,test)
    y_predictive_prob=np.mean(y_predictive_dist,axis=0)
    y_predictive=np.round(y_predictive_prob)
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))
    np.save("./npy_files/bayes_lda_zero.npy",{"posterior_samples":posterior_samples,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":NUM_CHAINS,"num_bayes_samples":NUM_BAYES_SAMPLES,"divergences":divergences})
