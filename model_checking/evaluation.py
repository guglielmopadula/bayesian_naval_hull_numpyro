import sys
import numpyro
import numpy as np
import sys
import jax.random as random
from numpyro import handlers
from pathlib import Path
from sklearn.metrics import confusion_matrix
import jax.numpy as jnp
from bayesian_methods.bayes_logreg_mle import model as bayes_logreg_mle
from bayesian_methods.bayes_logreg_zero import model as bayes_logreg_zero
from bayesian_methods.bayes_nb_mle import model as bayes_nb_mle
from bayesian_methods.bayes_nb_zero import model as bayes_nb_zero
from bayesian_methods.bayesian_nn_mle import model as bayes_nn_mle
from bayesian_methods.bayesian_nn_zero import model as bayes_nn_zero
from bayesian_methods.bayes_gam_zero import model as bayes_gam_zero
from bayesian_methods.bayes_gam_mle import model as bayes_gam_mle
from bayesian_methods.bayes_nbpp_zero import model as bayes_nbpp_zero
from bayesian_methods.bayes_lda_zero import model as bayes_lda_zero
from bayesian_methods.bayes_qda_zero import model as bayes_qda_zero

from bayesian_methods.bayes_logreg_mle import my_log_likehood as bayes_logreg_mle_lk
from bayesian_methods.bayes_logreg_zero import my_log_likehood as bayes_logreg_zero_lk
from bayesian_methods.bayes_nb_mle import my_log_likehood as bayes_nb_mle_lk
from bayesian_methods.bayes_nb_zero import my_log_likehood as bayes_nb_zero_lk
from bayesian_methods.bayesian_nn_mle import my_log_likehood as bayes_nn_mle_lk
from bayesian_methods.bayesian_nn_zero import my_log_likehood as bayes_nn_zero_lk
from bayesian_methods.bayes_gam_zero import my_log_likehood as bayes_gam_mle_lk
from bayesian_methods.bayes_gam_mle import my_log_likehood as bayes_gam_zero_lk
from bayesian_methods.bayes_nbpp_zero import my_log_likehood as bayes_nbpp_zero_lk
from bayesian_methods.bayes_lda_zero import my_log_likehood as bayes_lda_zero_lk
from bayesian_methods.bayes_qda_zero import my_log_likehood as bayes_qda_zero_lk

import arviz as az


##############REMEMBER TO DO export PYTHONPATH="${PYTHONPATH}:." && export XLA_PYTHON_CLIENT_MEM_FRACTION=.10

def compute_bayesian_p_value(y_fitted_dist,y):
    mean_dist=np.mean(y_fitted_dist,axis=-1)
    mean=np.mean(y)
    print("Mean pvalue is", np.mean(mean_dist>=mean))

def my_loo(posterior,train,target_train,likehood):
    likehood_data=likehood(posterior,train,target_train).reshape(-1,600)
    az_data = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior.items()},
    log_likelihood={"Y": likehood_data[None, ...]},)
    return az.loo(az_data,pointwise=True)  

def my_waic(posterior,train,target_train,likehood):
    likehood_data=likehood(posterior,train,target_train).reshape(-1,600)
    az_data = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior.items()},
    log_likelihood={"Y": likehood_data[None, ...]},)
    return az.waic(az_data,pointwise=True)  

def compute_posterior_average(posterior):
    posterior={k: jnp.mean(v,axis=(0,1)).reshape(1,*jnp.mean(v,axis=(0,1)).shape) for k, v in posterior.items()}
    return posterior



def compute_rse(posterior):
    l=[]
    for k,v in posterior.items():
        v=v.reshape(-1,*v.shape[2:])
        v=v.reshape(v.shape[0],-1)
        l.append(np.max(np.std(v[:,np.mean(v,axis=0)!=0],axis=0)/np.mean(v[:,np.mean(v,axis=0)!=0],axis=0)))
    return np.max(l)

def compute_rhat(posterior):
    l=[]
    for k,v in posterior.items():
        v=v.reshape(v.shape[0],v.shape[1],-1)
        l.append(np.max(np.nan_to_num(numpyro.diagnostics.split_gelman_rubin(v))))
    return np.max(l)

def my_dic(posterior,train,target_train,likehood):
    posterior_mean=compute_posterior_average(posterior)
    return 2*jnp.mean(jnp.sum(likehood(posterior_samples,train,target_train),axis=-1).reshape(-1))-jnp.sum(likehood(posterior_mean,train,target_train),axis=-1)

models_dict={}
models_dict["bayes_logreg_mle"]=bayes_logreg_mle
models_dict["bayes_logreg_zero"]=bayes_logreg_zero
models_dict["bayes_nb_mle"]=bayes_nb_mle
models_dict["bayes_nb_zero"]=bayes_nb_zero
models_dict["bayes_nbpp_zero"]=bayes_nbpp_zero
models_dict["bayesian_nn_mle"]=bayes_nn_mle
models_dict["bayesian_nn_zero"]=bayes_nn_zero
models_dict["bayes_gam_mle"]=bayes_gam_mle
models_dict["bayes_gam_zero"]=bayes_gam_zero
models_dict["bayes_lda_zero"]=bayes_lda_zero
models_dict["bayes_qda_zero"]=bayes_qda_zero


log_lik_dict={}
log_lik_dict["bayes_logreg_mle"]=bayes_logreg_mle_lk
log_lik_dict["bayes_logreg_zero"]=bayes_logreg_zero_lk
log_lik_dict["bayes_nb_mle"]=bayes_nb_mle_lk
log_lik_dict["bayes_nb_zero"]=bayes_nb_zero_lk
log_lik_dict["bayes_nbpp_zero"]=bayes_nbpp_zero_lk
log_lik_dict["bayesian_nn_mle"]=bayes_nn_mle_lk
log_lik_dict["bayesian_nn_zero"]=bayes_nn_zero_lk
log_lik_dict["bayes_gam_mle"]=bayes_gam_mle_lk
log_lik_dict["bayes_gam_zero"]=bayes_gam_zero_lk
log_lik_dict["bayes_lda_zero"]=bayes_lda_zero_lk
log_lik_dict["bayes_qda_zero"]=bayes_qda_zero_lk



NUM_SAMPLES=600




for name in ["bayes_logreg_zero","bayesian_nn_zero","bayes_nb_zero","bayes_nbpp_zero","bayes_gam_zero","bayes_lda_zero","bayes_qda_zero"]:
    data=np.load("npy_files/data.npy",allow_pickle=True).item()
    train=data["train"]
    test=data["test"]

    print(name)
    values=np.load("./npy_files/"+name+".npy",allow_pickle=True).item()
    sys.stdout=open("./model_checking/"+name+".txt","w+")
    posterior_samples=values["posterior_samples"]
    y_fitted_dist=values["y_fitted_dist"]
    y_predictive_dist=values["y_predictive_dist"]
    target_train=values["target_train"]
    target_test=values["target_test"]
    divergences=values["divergences"]
    NUM_CHAINS=values["num_chains"]
    NUM_BAYES_SAMPLES=values["num_bayes_samples"]
    y_fitted_prob=np.mean(y_fitted_dist,axis=0)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_dist=y_predictive_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_predictive_prob=np.mean(y_predictive_dist,axis=0)
    y_predictive=np.round(y_predictive_prob)
    numpyro.diagnostics.print_summary(posterior_samples,prob=0.95)
    compute_bayesian_p_value(y_fitted_dist,target_train)
    print("number of divergences is", divergences)
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))
    print("LOO")
    print("")
    print("")
    print(my_loo(posterior_samples,train,target_train,log_lik_dict[name]))
    print("WAIC")
    print("")
    print("")
    
    print("elpdWAIC is", my_waic(posterior_samples,train,target_train,log_lik_dict[name]))
    print("elpdDIC is",my_dic(posterior_samples,train,target_train,log_lik_dict[name]))
    print("rse is", compute_rse(posterior_samples))
    print("rhat is", compute_rhat(posterior_samples))


    sys.stdout.close()
    sys.stdout=sys.__stdout__


    
