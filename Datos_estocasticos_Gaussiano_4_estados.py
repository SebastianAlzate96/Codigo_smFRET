
"""
----------------------------------------------------------------------------
             Datos_estocasticos_Gaussiano_4_estados
----------------------------------------------------------------------------

"""

"""
----------------------------------------------------------------------------
                       # Paquetes #
----------------------------------------------------------------------------  
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from hmmlearn import hmm
from Definiciones import PoissonHMM
from Definiciones import gaussian_HMM_pseudo_residuals
# from Definiciones import duracion_rafaga
# import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

"""
----------------------------------------------------------------------------
             # Simulacion Estocastica HMM #
----------------------------------------------------------------------------  
"""

n_states = 4

# Medias para el background
m1 = 60 #nivel de fotones esperados del transfondo captados por el detectro de la aceptora
m2 = 90 #nivel de fotones esperados del transfondo captados por el detectro de la donante

ef1 = 0.30 #Eficiencia para el estado 1 FRET
ef2 = 0.50 #Eficiencia para el estado 2 FRET
ef3 = 0.70 #Eficiencia para el estado 3 FRET

m1a =  10.10 # media ajustada para la aceptora en el primer estado
m1d = m1a*(1-ef1)/ef1 # media ajustada para la donante en el primer estado
                      # se hace de esa manera para que de la eficiencia deseada

m2a = (m1a+m1d)*ef2   # media ajustada para la donante en el segundo estado,
                      # se hace de esta forma para m1a+m1d=m2a+m2d
m2d = m2a*(1-ef2)/ef2 # media ajustada para la donante en el primer estado
                      # se hace de esa manera para que de la eficiencia deseada

m3a = (m2a+m2d)*ef3   # media ajustada para la donante en el segundo estado,
                      # se hace de esta forma para m1a+m1d=m2a+m2d
m3d = m3a*(1-ef3)/ef3 # media ajustada para la donante en el primer estado
                      # se hace de esa manera para que de la eficiencia deseada

# Matriz de medias, primer estado es el background, 
# los otros dos el estado FRET mas las tasa del background
means = np.array([[m1, m2],
                  [m1a+m1, m1d+m2],
                  [m2a+m1, m2d+m2],
                  [m3a+m1, m3d+m2]])

#print('medias reales:',np.round(means,2))


# Probabilidad inicial
startprob = np.array([1, 0, 0, 0])


# Matriz de transicion 
transmat = np.array([[0.60,0.13,0.13,0.14],
                     [0.24,0.40,0.18,0.18],
                     [0.25,0.15,0.45,0.15],
                     [0.20,0.15,0.15,0.50]])



# Matriz de transicion ajustada, hay que eliminar
# primera fila y columna que es la del background
a=np.delete(transmat, 0,0)
gamma_=np.delete(a, 0,1)
for i in range(n_states-1):
  gamma_[i,:]=gamma_[i,:]/(np.sum(gamma_,axis=1)[i]) 
  
print("\n \n-----Matriz de transicion Ajustada Real-----\n")
print(np.round(gamma_,2))

# Construir modelo con los parametros predefinidos
np.random.seed(1)
gen_model = PoissonHMM(n_components=n_states,n_iter=1000)

# parámetros, las medias de los componentes
gen_model.startprob_ = startprob
gen_model.transmat_ = transmat
gen_model.means_ = means

# Generar muestra
X, Z = gen_model.sample(30000) # X es la muestra y Z el camino Viterbi


"""
----------------------------------------------------------------------------
             #  Valores estimados modelo  Gaussiano HMM#
----------------------------------------------------------------------------  
"""

markovmodel_P = hmm.GaussianHMM(n_components=n_states,n_iter=1000,covariance_type="full")


markovmodel_P.fit(X) #Estima los parámetros del modelo

print(' \n Numero de iteraciones = ', markovmodel_P.monitor_.iter) #esto es el número de iteraciones que tomó para que EM convergiera
print('\n Convergencia = ', markovmodel_P.monitor_.converged) # me dice si hay convergencia

means_ = markovmodel_P.means_ # Valores esperados estimados

markovmodel_P.transmat_ # Matriz de transicion estimada

# Organizar medias de menor a mayor
means_sort = np.sort(means_[:,0])
means_2 = np.zeros((n_states, 2)).astype(float)

for i in range(n_states):
  means_2[i,:]=means_[list(means_[:,0]).index(means_sort[i]),:]

#Varianzas empiricas
states= markovmodel_P.predict(X) # Camino Viterbi

variance_ = np.empty([n_states,2])
for k in range(2):
  for s in range(n_states):
    y=X[:,k][states==list(means_[:,0]).index(means_sort[s])]
    variance_[s,k] = y.var()

con_ = np.concatenate((means_2,variance_),axis=1)

row_indices = ["Background", "State 1", "State 2", "State 3"]
column_names = ["Mean Acceptor", "Mean Donor","Acceptor Variance","Donor Variance"]
means_ = pd.DataFrame(np.round(con_,2), index=row_indices, columns=column_names)

print("\n \n-----Parametros Estiamdos-----\n")
print(means_) # imprime los valores esperados estimados y la varianza
              # empiricas en cada estado y ordenado de forma ascedente



"""
----------------------------------------------------------------------------
                          # Modelo Ajustado #
----------------------------------------------------------------------------  
"""

# Valores esperados ajustados, restar la tasa del trasfondo
# que es la mas pequena
means_ad = means_2-means_2[0,:]
means_ad = np.delete(means_ad,0,0)

row_indices = ["Estado 1", "Estado 2", "Estado 3"]
column_names = ["Aceptor", "Donante"]
means_ = pd.DataFrame(np.round(means_ad,2), index=row_indices, columns=column_names)
print("\n \n-----Parametros Esperados Estimados Ajustados-----\n")
print(means_) # valores esperados ajustados y organizados segun la eficiencia


# Eficiencias para los estados FRET
eff=means_ad[:,0]/(means_ad[:,0]+means_ad[:,1])
row_indices = ["Estado 1", "Estado 2", "Estado 3"]
column_names = ["Eficiencias"]
eff_ = pd.DataFrame(np.round(eff,2), index=row_indices, columns=column_names)
print("\n \n-----Eficiencias Estimadas-----\n")
print(eff_)


markovmodel_P.means_=markovmodel_P.means_
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
    
transmat_ = pd.DataFrame( np.round(gamm_3,2))
print("\n \n-----Matriz Estimada de transicion Ajustada-----\n")
print(transmat_, '\n')

"""
-------------------------------------------------------------------------------
          # Modelo univariado de 2 estados (para detectar rafagas) #
-------------------------------------------------------------------------------  
"""
# # sumar los conteos de la aceptora y donante en cada tiempo
# Y = np.zeros([len(X),1])
# Y[:,0]=X[:,0]+X[:,1]


# markovmodel_P=hmm.GaussianHMM(n_components=2,n_iter=1000,covariance_type="full")
# markovmodel_P.fit(Y) #Estima los parámetros del modelo

# mea = np.round(markovmodel_P.means_,2)
# print(np.round(mea,2))
# tra = np.round(markovmodel_P.transmat_,2)
# print(np.round(tra,2))


"""
-------------------------------------------------------------------------------
            #  Duracion de rafagas en el camino viterbi #
-------------------------------------------------------------------------------  
"""

# markovmodel_P= hmm.GaussianHMM(n_components=4, n_iter=1000, covariance_type="full")

# duracion_rafaga(markovmodel_P, Y)


"""
-------------------------------------------------------------------------------
                        #  Graficas #
-------------------------------------------------------------------------------  
"""

input("Graficas: si=1 no=0")
for i in range(0,1):
    if (input() != "1"):
        print("Fin")
        break
    else:
        """
        ## Histogramas de conteos de la aceptora y donante
        """
        
        t = 800  # Definir el punto de tiempo
        fig, ax = plt.subplots()
        lista = list(range(t))
        datos = {'Acceptor': X[:t, 0], 'Donor': X[:t, 1]}
        ax.plot(lista, datos['Acceptor'], label='Acceptor')
        ax.plot(lista, datos['Donor'], label='Donor')
        ax.legend(loc='upper left')
        ax.set_xlabel("Tiempo")
        
        #   # Especificar la ruta completa al directorio de destino
        # directorio_destino = r'C:\Users\sebas\Desktop\Enzimas\Datos_simulados\Gaussiano_4_estados'
        
        # # Verificar si el directorio existe y, si no, crearlo
        # if not os.path.exists(directorio_destino):
        #     os.makedirs(directorio_destino)
        
        # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'his_Gaussiano_acep_'+str(m1)+'.0_don_'+str(m2)+'.0.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        plt.show()
        
        
        """
        ## Histogramas para la suma de conteos de la aceptora y donante
        """
        
        # # t=800 # definir el punto de tiempo
        # # fig, ax = plt.subplots()
        # # lista = list(range(t))
        # # datos = {'Total':X[:t,0]+X[:t,1]}
        # # ax.plot(lista, datos['Total'], label = 'Conteo total')
        # # ax.legend(loc = 'upper left')
        # # ax.set_xlabel("Tiempo")
        # # #plt.savefig('his_total_Gaussian_acep_'+str(m1)+'.0_don_'+str(m2)+'.png', bbox_inches='tight', transparent=True)
        # # plt.show()
        
        
        """
        ## Camino Viterbi para 3 estados
        """
        
        # n_states=3
        # markovmodel_P = hmm.GaussianHMM(n_components=n_states,n_iter=1000,covariance_type="full")
        # markovmodel_P.fit(X) #Estima los parámetros del modelo
        
        # states= markovmodel_P.predict(X) #Encuentre la secuencia de estado más probable correspondiente a X.
        
        # # Camino Viterbi
        # markovmodel_P.means_=markovmodel_P.means_
        # lam_=markovmodel_P.means_[:,0][states[:t]]
        # lam=list(markovmodel_P.means_[:,0])
        # list_=list(np.sort(lam))
  
        # # Organizar para que el estado cero sea el background
        # for i in range(n_states):
        #   lam_[lam_==list_[i]]=i
          
  
        # fig, ax = plt.subplots()
        # ax.plot(lam_, color = 'green')
        # ax.set_ylabel('Estado')
        # ax.set_xlabel('Tiempo')
        # plt.yticks(range(0,4,1))
  
        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'viterbi_3_states_Gaussiano_acep_'+str(m1)+'.0_don_'+str(m2)+'.0.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()   
        
        """
        ## Camino Viterbi 4 estados
        """
        
        n_states=4
        np.random.seed(1)
        
        markovmodel_P = hmm.GaussianHMM(n_components=n_states,n_iter=1000,covariance_type="full",random_state=3)
        markovmodel_P.fit(X) #Estima los parámetros del modelo
        
        states= markovmodel_P.predict(X) #Encuentre la secuencia de estado más probable correspondiente a X.
        
        # Camino Viterbi
        markovmodel_P.means_=markovmodel_P.means_
        lam_=markovmodel_P.means_[:,0][states[:t]]
        lam=list(markovmodel_P.means_[:,0])
        list_=list(np.sort(lam))
        
        # Organizar para que el estado cero sea el background
        for i in range(n_states):
          lam_[lam_==list_[i]]=i
        
        fig, ax = plt.subplots()
        ax.plot(lam_, color = 'green')
        ax.set_ylabel('Estado')
        ax.set_xlabel('Tiempo')
        plt.yticks(range(0,4,1))
        #plt.savefig('viterbi_4_states_Gaussian_acep_'+str(m1)+'.0_don_'+str(m2)+'.png', bbox_inches='tight', transparent=True)
        plt.show()
        
        """
        ## Camino Viterbi 5 estados
        """
        
        # # n_states=5
        # # np.random.seed(1)
        
        # # markovmodel_P = hmm.GaussianHMM(n_components=n_states,n_iter=1000,covariance_type="full",random_state=3)
        # # markovmodel_P.fit(X) #Estima los parámetros del modelo
        
        # # states= markovmodel_P.predict(X) #Encuentre la secuencia de estado más probable correspondiente a X.
        
        # # # Camino Viterbi
        # # markovmodel_P.means_=markovmodel_P.means_
        # # lam_=markovmodel_P.means_[:,0][states[:t]]
        # # lam=list(markovmodel_P.means_[:,0])
        # # list_=list(np.sort(lam))
        
        # # # Organizar para que el estado cero sea el background
        # # for i in range(n_states):
        # #   lam_[lam_==list_[i]]=i
          
        # # fig, ax = plt.subplots()
        # # ax.plot(lam_, color = 'green')
        # # ax.set_ylabel('Estado')
        # # ax.set_xlabel('Tiempo')
        # # #plt.savefig('viterbi_5_states_Gaussian_acep_'+str(m1)+'.0_don_'+str(m2)+'.png', bbox_inches='tight', transparent=True)
        # # plt.show()
        
        
        # Crear una figura con una sola fila y cuatro columnas
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for h in range(4):  # Iterar sobre las cuatro gráficas
        
            markovmodel=hmm.GaussianHMM(n_components=4, covariance_type='full',n_iter=1000)
            markovmodel.fit(X)
            lamb_=markovmodel.means_
            gamma_=markovmodel.transmat_
            delta_=markovmodel.get_stationary_distribution()
            cov_matrix = markovmodel.covars_
            
            std_matrix = np.sqrt(np.array([np.diag(matrix) for matrix in cov_matrix]))
        
            # Seleccionar el eje actual para trazar
            ax = axes[h]
        
            if h == 0:
                res_acep = gaussian_HMM_pseudo_residuals(x=X[:, 0], delta=delta_, gamma=gamma_, mu=lamb_[:,0], sigma=std_matrix[:,0], m=4)
                x = np.arange(-4, 4, 0.001)
                intervalos = range(-6, 7)
                ax.plot(x, norm.pdf(x, 0, 1))
                ax.hist(res_acep[0], color='gray', bins=intervalos, rwidth=0.85, density=True)
                ax.set_title('Aceptora Normal')
            elif h == 1:
                res_acep = gaussian_HMM_pseudo_residuals(x=X[:, 0], delta=delta_, gamma=gamma_, mu=lamb_[:,0], sigma=std_matrix[:,0], m=4)
                ax.hist(x=res_acep[1], color='gray', rwidth=0.5, density=True)
                ax.set_title('Aceptora Uniforme')
                ax.axhline(y=1)
            elif h == 2:
                res_dono = gaussian_HMM_pseudo_residuals(x=X[:, 1], delta=delta_, gamma=gamma_, mu=lamb_[:,1], sigma=std_matrix[:,1], m=4)
                x = np.arange(-4, 4, 0.001)
                intervalos = range(-6, 7)
                ax.plot(x, norm.pdf(x, 0, 1))
                ax.hist(res_dono[0], color='gray', bins=intervalos, rwidth=0.85, density=True)
                ax.set_title('Donante Normal')
            elif h == 3:
                res_dono = gaussian_HMM_pseudo_residuals(x=X[:, 1], delta=delta_, gamma=gamma_, mu=lamb_[:,1], sigma=std_matrix[:,1], m=4)
                ax.hist(x=res_dono[1], color='gray', rwidth=0.5, density=True)
                ax.set_title('Donante Uniforme')
                ax.axhline(y=1)
        
        plt.tight_layout()
        # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'Pseudo_residuales_Gaussiano_acep_'+str(m1)+'.0_don_'+str(m2)+'.0.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        plt.show()

        
        """
        ## Grafica del *BIC*
        """
        
        scores=list()
        
        for n_comp in range(2,11):
          model = hmm.GaussianHMM(n_components=n_comp,n_iter=1000, covariance_type='full',tol=1e-5)
          model.fit(X)
          c = n_comp*n_comp+2*n_comp-1
          print(model.score(X))
          scores.append(c*np.log(len(X[:,0]))-2*model.score(X))
        
        print(np.round(scores,2))
        fig, ax = plt.subplots()
        lista = [2,3,4,5,6,7,8,9,10]
        datosbic = {'traye1':scores}
        ax.plot(lista, datosbic['traye1'])
        ax.set_xlabel("Num. Estados")
        ax.set_ylabel("BIC")
        ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed')

        # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'BIC_Gaussiano_acep_'+str(m1)+'.0_don_'+str(m2)+'.0.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        plt.show()
        
        porcen = list()

        for k in range(len(scores)-1):
            porcen.append((scores[k]-scores[k+1])/scores[k]*100)
            
        print(np.round(porcen,2))

