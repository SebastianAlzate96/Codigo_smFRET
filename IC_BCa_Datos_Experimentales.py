
"""
----------------------------------------------------------------------------
             IC BCa Datos experimentales
----------------------------------------------------------------------------

"""

import pandas as pd
import numpy as np
# from Definiciones import PoissonHMM
import scipy.stats as st
from hmmlearn import hmm
from joblib import Parallel, delayed
import time
import warnings
# from sklearn.utils import check_random_state
warnings.filterwarnings("ignore", category=FutureWarning)

# Cargar los datos experimentales, estos contienen 10 condiciones de 
# desnaturalizante y 4 trayectorias

"""
----------------------------------------------------------------------------
             # Definiciones #
----------------------------------------------------------------------------  
"""

def statfunction_means(data, rg = None):
    
    if rg is None:
        rg = np.random.RandomState()
        
    n =len(data)
    
    x_star = markovmodel_P.sample(n_samples=n, random_state=rg)[0]
    
    markovmodel_ = hmm.GaussianHMM(n_components=n_states,n_iter=200,covariance_type="full",tol=1e-5)
    markovmodel_.fit(x_star) #Estima los parámetros del modelo
    means_ = markovmodel_.means_
    
    means_sort = np.sort(means_[:,0])
    means_2 = np.zeros((n_states, 2)).astype(float)

    for i in range(n_states):
      means_2[i,:]=means_[list(means_[:,0]).index(means_sort[i]),:]
      
    means_ad = means_2-means_2[0,:]
    means_ad = np.delete(means_ad,0,0)
    means_ad_ = means_ad.flatten()
    
    min_=np.amin(markovmodel_.means_[:,0])
    posi=np.where(markovmodel_.means_[:,0] == min_)
    min2_=markovmodel_.means_[posi[0][0],1]
    means_ad=np.delete(markovmodel_.means_, posi,0)-np.array([min_,min2_])
    means_sort_ad = np.sort(means_ad[:,0])
    
    a=np.delete(markovmodel_.transmat_, posi,0)
    gamm_=np.delete(a, posi,1)
    for i in range(n_states-1):
      gamm_[i,:]=gamm_[i,:]/(np.sum(gamm_,axis=1)[i]) 
    
    gamm_2=np.zeros((n_states-1, n_states-1)).astype(float)
    for i in range(n_states-1):
        gamm_2[i,:]=gamm_[list(means_ad[:,0]).index(means_sort_ad[i]),:]
        
    gamm_3=np.zeros((n_states-1, n_states-1))
    gamm_3=gamm_3.astype(float)
    for i in range(n_states-1):
        gamm_3[:,i]=gamm_2[:,list(means_ad[:,0]).index(means_sort_ad[i])]
            
    gamm_3=gamm_3.flatten()
    
    parametros = np.concatenate((means_ad_, gamm_3))
      
    return parametros

# statfunction_data = statfunction_means(X)


def bootstrap_replicates(data,num_simu):
    
    def draw_bs_sample(data, rg = None):
        sample_indices = statfunction_means(data, rg = None)
        return sample_indices
    
    resamples = Parallel(n_jobs=5)(delayed(draw_bs_sample)(data) for _ in range(num_simu))
    boot_reps = np.vstack(resamples)
    
    return boot_reps

# boot_reps = bootstrap_replicates(X,70)

def compute_z0(data, boot_reps, statfunction=statfunction_means):
    '''Computes z0 for given data and statistical function'''
    s = np.empty((n_states**2-1))
    
    for i in range(n_states**2-1):
        # print('z0 ', i)
        s_ = parametros_estimados[i]
        s[i] = st.norm.ppf(np.sum(boot_reps[:,i] < s_) / len(boot_reps[:,i]))

    return s

# compute_z0(X, boot_reps)

def jackknife_parallel(data, statfunction=statfunction_means, n_jobs=4):
    n = len(data)
    
    def compute_jackknife_reps(data, statfunction , i):
        jack_sample = np.delete(data, i,0)
        
        return statfunction(jack_sample)
        
    jk_estimates = Parallel(n_jobs=n_jobs)(delayed(compute_jackknife_reps)(data, statfunction_means, i) for i in range(n))
    jack_reps = np.array(jk_estimates)
    
    return jack_reps

# jack_reps = jackknife_parallel(X)


def compute_a(jack_reps):
    '''Returns the acceleration constant a'''
    a = np.empty((n_states**2-1))
    
    for i in range(n_states**2-1):
        # print('jack_reps ',i)
        mean = np.mean(jack_reps[:,i])
        a[i] = (1/6) * np.divide(np.sum(mean - jack_reps[:,i])**3, (np.sum(mean - jack_reps[:,i])**2)**(3/2))
    return a

# compute_a(jack_reps)


def compute_bca_ci(data, alpha_level, num_simu, statfunction=statfunction_means):
    '''Returns BCa confidence interval for given data at given alpha level'''
    # Compute bootstrap and jackknife replicates
    boot_reps = bootstrap_replicates(data, num_simu)
    jack_reps = jackknife_parallel(data, statfunction)

    # Compute a and z0
    a = compute_a(jack_reps)
    z0 = compute_z0(data, boot_reps)

    # Compute confidence interval indices
    alphas = np.array([alpha_level/2., 1-alpha_level/2.])
    zs = z0 + st.norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
    avals = st.norm.cdf(z0 + zs/(1-a*zs))
    ints = np.round((len(boot_reps)-1)*avals)
    ints = np.nan_to_num(ints).astype('int')

    # Compute confidence interval
    ci_low = np.empty((n_states**2-1))
    ci_high = np.empty((n_states**2-1))
    
    for i in range(n_states**2-1):
        boot_reps_ = np.sort(boot_reps[:,i])
        ci_low[i] = boot_reps_[ints[0][i]]
        ci_high[i] = boot_reps_[ints[1][i]]
        
        
    return (ci_low, ci_high)


           
"""
----------------------------------------------------------------------------
             # Datos experimentales #
----------------------------------------------------------------------------  
"""
# Condiciones de desnaturalizante
cond=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0]
c = 0

# Tratectorias
trayec=[1,2,3,4]
tr = 0

    
print("IC_BCa_Gaussiano_3_estados_condicion_" + str(cond[c])+ "_trayectoria_" + str(trayec[tr]))

inicio = time.time()

# Cambiar ruta
filename = '/home/eflorez/salzate/Tesis_Enzimas/data_all.csv'
# filename='C:/Users/sebas/Desktop/Enzimas/data_all.csv'
df=pd.read_csv(filename,sep=",")
new_column_names = ['num', 'Time', 'Acceptor', 'donor', 'condition', 'trayectory']
df.columns = pd.Index(new_column_names)


df = df[df['condition'] == cond[c]] # escoger una condicion de desnaturalizante
df = df[df['trayectory'] == trayec[tr]] # escoger una trayectoria

# Organizar los datos en una matriz
X=np.zeros([len(df['Acceptor']),2])
X[:,0]=df['Acceptor']
X[:,1]=df['donor']

# X = X[:500]
"""
----------------------------------------------------------------------------
             # Modelo AMHMM #
----------------------------------------------------------------------------  
"""

n_states = 3

markovmodel_P= hmm.GaussianHMM(n_components=n_states,n_iter=200,covariance_type="full",tol=1e-5)

markovmodel_P.fit(X) #Estima los parámetros del modelo

means_ = markovmodel_P.means_

means_sort = np.sort(means_[:,0])
means_2 = np.zeros((n_states, 2)).astype(float)

for i in range(n_states):
  means_2[i,:]=means_[list(means_[:,0]).index(means_sort[i]),:]
  
means_ad = means_2-means_2[0,:]
means_ad = np.delete(means_ad,0,0)
means_ad_ = means_ad.flatten()

min_=np.amin(markovmodel_P.means_[:,0])
posi=np.where(markovmodel_P.means_[:,0] == min_)
min2_=markovmodel_P.means_[posi[0][0],1]
means_ad=np.delete(markovmodel_P.means_, posi,0)-np.array([min_,min2_])
means_sort_ad = np.sort(means_ad[:,0])

a=np.delete(markovmodel_P.transmat_, posi,0)
gamm_=np.delete(a, posi,1)
for i in range(n_states-1):
  gamm_[i,:]=gamm_[i,:]/(np.sum(gamm_,axis=1)[i]) 

gamm_2=np.zeros((n_states-1, n_states-1)).astype(float)
for i in range(n_states-1):
    gamm_2[i,:]=gamm_[list(means_ad[:,0]).index(means_sort_ad[i]),:]
    
gamm_3=np.zeros((n_states-1, n_states-1))
gamm_3=gamm_3.astype(float)
for i in range(n_states-1):
    gamm_3[:,i]=gamm_2[:,list(means_ad[:,0]).index(means_sort_ad[i])]
        
gamm_3=gamm_3.flatten()

parametros_estimados = np.concatenate((means_ad_, gamm_3))
    
    
    
    
            
"""
----------------------------------------------------------------------------
             #  Crear IC #
----------------------------------------------------------------------------  
"""

BCa = compute_bca_ci(data = X, alpha_level = 0.05, num_simu = 50000)

resultados_df = pd.DataFrame({
    "conf_int_low": BCa[0],
    "conf_int_high": BCa[1],
    "Difference": BCa[1]-BCa[0]
})

resultados_df = np.round(resultados_df,3)

fn = "IC_BCa_Gaussiano_3_estados_condicion_" + str(cond[c])+ "_trayectoria_" + str(trayec[tr]) +".csv"

resultados_df.to_csv(fn, index=False)

print(resultados_df)

fin = time.time()
print(fin-inicio)  
    
