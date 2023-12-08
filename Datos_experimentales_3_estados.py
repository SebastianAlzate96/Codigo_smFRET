"""
----------------------------------------------------------------------------
             Datos Experimentales 3 estados
----------------------------------------------------------------------------

"""

"""
----------------------------------------------------------------------------
                       # Paquetes #
----------------------------------------------------------------------------  
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
# from scipy.stats import norm
# from Definiciones import gaussian_HMM_pseudo_residuals
# from Definiciones import duracion_rafaga
from sklearn.utils import check_random_state
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""
----------------------------------------------------------------------------
                    # Datos Experimentales #
----------------------------------------------------------------------------  
"""

# Cargar los datos experimentales, estos contienen 10 condiciones de 
# desnaturalizante y 4 trayectorias

# Condiciones de desnaturalizante
cond=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0]

# Tratectorias
trayec=[1,2,3,4]


for j in range(len(trayec)):
    for l in range(len(cond)):
        
        # np.random.seed(3)
        
        # rs = check_random_state(1)

        # Cambiar ruta
        filename='C:/Users/sebas/Desktop/Enzimas/data_all.csv'
        df=pd.read_csv(filename,sep=",")
        new_column_names = ['num', 'Time', 'Acceptor', 'donor', 'condition', 'trayectory']
        df.columns = pd.Index(new_column_names)
        

        df = df[df['condition'] == cond[l]] # escoger una condicion de desnaturalizante
        df = df[df['trayectory'] == trayec[j]] # escoger una trayectoria
        
        # Organizar los datos en una matriz
        X=np.zeros([len(df['Acceptor']),2])
        X[:,0]=df['Acceptor']
        X[:,1]=df['donor']
        
        print('\n ****** Trayectoria '+str(trayec[j])+' condicion '+str(cond[l])+' ****** \n')
        
        
        """
        ----------------------------------------------------------------------------
                     #  Modelo Gaussiano HMM #
        ----------------------------------------------------------------------------  
        """
        
        n_states=3
        
        best_score = float("-inf")  # Inicializa el mejor puntaje con un valor negativo infinito
        best_seed = None  # Inicializa la mejor semilla como None

        # for k in range(7):
        #     # rs = np.random.RandomState()
            
        #     rs = check_random_state(k)
            
        #     markovmodel = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, tol=1e-5, random_state=rs)
        #     markovmodel.fit(X)  # Estima los parámetros del modelo
            
        #     # print(markovmodel_P.monitor_.iter) #esto es el número de iteraciones que tomó para que EM convergiera
        #     # print(markovmodel_P.monitor_.converged) # me dice si hay convergencia

        #     score = markovmodel.score(X)

        #     # print(f'Score {i}: {score}')

        #     # Comprueba si este score es mejor que el anterior
        #     if score > best_score:
        #         best_score = score
        #         best_seed = k
    
        # rs_ = check_random_state(best_seed)
        
        
        markovmodel_P = hmm.GaussianHMM(n_components=n_states, n_iter=1000, covariance_type="full")
        
        markovmodel_P.fit(X) #Estima los parámetros del modelo
        
        # print('Numero de iteraciones: ', markovmodel_P.monitor_.iter) #esto es el número de iteraciones que tomó para que EM convergiera
        # print('Converge?', markovmodel_P.monitor_.converged)
        
        # # lines = inspect.getsource(PoissonHMM)
        # # print(lines)
        # print(markovmodel_P.covars_)
        
        """
        ----------------------------------------------------------------------------
                      #  Valores estimados modelo  Gaussiano HMM #
        ----------------------------------------------------------------------------  
        """
        
        """
        ## Valores esperados
        """
        
        means_ = markovmodel_P.means_ 
        
        # Organizar medias de menor a mayor
        means_sort = np.sort(means_[:,0])
        means_2 = np.zeros((n_states, 2)).astype(float)
        
        for i in range(n_states):
          means_2[i,:]=means_[list(means_[:,0]).index(means_sort[i]),:]
        
        
        """
        ## Varianzas empiricas
        """
        states= markovmodel_P.predict(X) # Camino Viterbi
        
        variance_ = np.empty([n_states,2])
        for k in range(2):
          for s in range(n_states):
            y=X[:,k][states==list(means_[:,0]).index(means_sort[s])]
            variance_[s,k] = y.var()
        
        con_ = np.concatenate((means_2,variance_),axis=1)
        
        row_indices = ["Background", "State 1", "State 2"]
        column_names = ["Mean Acceptor", "Mean Donor","Acceptor Variance","Donor Variance"]
        means__ = pd.DataFrame(np.round(con_,2), index=row_indices, columns=column_names)
        
        print("\n \n-----Parametros Estiamdos-----\n")
        print(means__)
        
        
        """
        ----------------------------------------------------------------------------
                                  # Modelo Ajustado #
        ----------------------------------------------------------------------------  
        """
        # Valores esperados ajustados, restar la tasa del trasfondo
        # que es la mas pequena
        means_T = np.transpose(means_2)
        means_adj = means_2-means_T[:,0]
        means_ad = np.delete(means_adj,0,0)
        
        row_indices = ["Estado 1", "Estado 2"]
        column_names = ["Aceptor", "Donante"]
        means_ad_ = pd.DataFrame(np.round(means_ad,2), index=row_indices, columns=column_names)
        
        print("\n \n-----Parametros Esperados Estimados Ajustados-----\n")
        print(means_ad_)
        
        # Eficiencias para los estados FRET
        eff=means_ad[:,0]/(means_ad[:,0]+means_ad[:,1])
        row_indices = ["Estado 1", "Estado 2"]
        column_names = ["Eficiencias"]
        eff_ = pd.DataFrame(np.round(eff,2), index=row_indices, columns=column_names)
        
        print("\n \n-----Eficiencias Estimadas-----\n")
        print(eff_)
        
        
        # Matriz de transicion
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
        print(transmat_)
        
        print("\n \n-------------------------------------------------------------------------------\n")
        
        
        """
        -------------------------------------------------------------------------------
                 # Modelo univariado de 2 estados (para detectar rafagas) #
        -------------------------------------------------------------------------------  
        """
        # # sumar los conteos de la aceptora y donante en cada tiempo
        # Y = np.zeros([len(X),1])
        # Y[:,0]=X[:,0]+X[:,1]
        
        
        # markovmodel_P=hmm.GaussianHMM(n_components=2,n_iter=1000,covariance_type="full", random_state=rs)
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
        
        # markovmodel_P= hmm.GaussianHMM(n_components=3, n_iter=1000, covariance_type="full", random_state=rs)
        
        # duracion_rafaga(markovmodel_P, Y)
        
        
        """
        -------------------------------------------------------------------------------
                                #  Graficas #
        -------------------------------------------------------------------------------  
        """
        
        # # Especificar la ruta completa al directorio de destino
        # directorio_destino = r'C:/Users/sebas/Desktop/Enzimas/Datos_experimentales/Histograma'
         
        # # Verificar si el directorio existe y, si no, crearlo
        # if not os.path.exists(directorio_destino):
        #     os.makedirs(directorio_destino)
        
        # """
        # ## Histogramas de conteos de la aceptora y donante
        # """
        
        t=800 # definir el punto de tiempo
        # fig, ax = plt.subplots()
        # lista = list(range(t))
        # datos = {'Acceptor':X[:t,0],'Donor':X[:t,1]}
        # ax.plot(lista, datos['Acceptor'], label = 'Acceptor')
        # ax.plot(lista, datos['Donor'], label = 'Donor')
        # ax.legend(loc = 'upper left')
        # ax.set_xlabel("Tiempo")
        
        
        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'his_condicion_'+str(cond[l])+'_trayectoria_'+str(trayec[j])+'.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()
        
        """
        -------------------------------------------------------------------------------
                    #  Histogramas #
        -------------------------------------------------------------------------------  
        """
        
        # suma_canales = X[:, 0] + X[:, 1]
        
        # # Especificar la ruta completa al directorio de destino
        # directorio_destino = r'C:/Users/sebas/Desktop/Enzimas/Datos_experimentales/Histograma_suma'
        
        # # Crear histograma para la Suma de Canal_Aceptora y Canal_Donante
        # plt.hist(suma_canales, bins=20, alpha=0.5, label='Suma Canales', rwidth=0.9, color='black')
        # plt.xlabel('Número de fotones')
        # plt.ylabel('Frecuencia')
        # # plt.legend()
        # plt.title('Histograma para total de fotones, condicion: '+str(cond[l])+' y trayectoria '+str(trayec[j]))

        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'his_suma_umbral_Gaussiano_3_states_condicion_'+str(cond[l])+'_trayectoria_'+str(trayec[j])+'.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()
        
        
        """
        ## Histogramas para la suma de conteos de la aceptora y donante
        """
        
        # t=800 # definir el punto de tiempo
        # fig, ax = plt.subplots()
        # lista = list(range(t))
        # datos = {'Total':X[:t,0]+X[:t,1]}
        # ax.plot(lista, datos['Total'], label = 'Conteo total')
        # ax.legend(loc = 'upper left')
        # ax.set_xlabel("Tiempo")
        # #plt.savefig('his_total_condicion_0.0_trayectoria_1.png', bbox_inches='tight', transparent=True)
        # plt.show()
        
        
        """
        ## Camino Viterbi para 3 estados
        """
        
        # # Especificar la ruta completa al directorio de destino
        # directorio_destino = r'C:/Users/sebas/Desktop/Enzimas/Datos_experimentales/Viterbi'
         
        # # Verificar si el directorio existe y, si no, crearlo
        # if not os.path.exists(directorio_destino):
        #     os.makedirs(directorio_destino)
        
        # states = markovmodel_P.predict(X) #Encuentre la secuencia de estado más probable correspondiente a X.
        
        # # Camino Viterbi
        
        # lam_ = markovmodel_P.means_[:,0][states[:t]]
        # lam = list(markovmodel_P.means_[:,0])
        # list_ = list(np.sort(lam))
        
        # # Organizar para que el estado cero sea el background
        # for i in range(n_states):
        #   lam_[lam_ == list_[i]] = i
          
        
        # fig, ax = plt.subplots()
        # ax.plot(lam_, color = 'green')
        # ax.set_ylabel('Estado')
        # ax.set_xlabel('Tiempo')
        # plt.yticks(range(0,3,1))
        
        
        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'viterbi_3_states_condicion_'+str(cond[l])+'_trayectoria_'+str(trayec[j])+'.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()
        
        
        """
        ## Pseudo Residuales
        """
        
        # # Especificar la ruta completa al directorio de destino
        # directorio_destino = r'C:/Users/sebas/Desktop/Enzimas/Datos_experimentales/Pseudo_Residuales'
         
        # # Verificar si el directorio existe y, si no, crearlo
        # if not os.path.exists(directorio_destino):
        #     os.makedirs(directorio_destino)
            
        # n_states = 3
        
        # markovmodel=hmm.GaussianHMM(n_components=n_states,n_iter=1000, covariance_type="full", random_state=rs)
        # markovmodel.fit(X)
        # lamb_=markovmodel.means_
        # gamma_=markovmodel.transmat_
        # delta_=markovmodel.get_stationary_distribution()
        # cov_matrix = markovmodel.covars_
        
        # std_matrix = np.sqrt(np.array([np.diag(matrix) for matrix in cov_matrix]))
        
        
        # """
        # ## Pseudo Residuales -- Aceptora Normal
        # """
        
        # res_acep = gaussian_HMM_pseudo_residuals(x=X[:,0], delta=delta_, gamma=gamma_, mu=lamb_[:,0], sigma=std_matrix[:,0], m=n_states)
            
        # fig, ax = plt.subplots()
        # x = np.arange(-4, 4, 0.001)
        # intervalos = range(-6, 7)
        # ax.plot (x, norm.pdf (x, 0, 1))
        # ax.hist(res_acep[0], color='gray', bins=intervalos,rwidth=0.85,density=True)
        # ax.set_title('Histograma pseudo residuales normales aceptora', loc = "center", fontdict = {'fontsize':11, 'fontweight':'normal', 'color':'black'})
        
        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'Aceptora_N_condicion_'+str(cond[l])+'_trayectoria_'+str(trayec[j])+'.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()
        
        
        # """
        # ## Pseudo Residuales -- Donante Normal
        # """
        
        # res_dono=gaussian_HMM_pseudo_residuals(x=X[:,1], delta=delta_, gamma=gamma_, mu=lamb_[:,1], sigma=std_matrix[:,1], m=n_states)
            
        # #para obtener los graficos de la normal, descomentar las primeras tres lineas
        # fig, ax = plt.subplots()
        # x = np.arange(-4, 4, 0.001)
        # intervalos = range(-6, 7)
        # ax.plot (x, norm.pdf (x, 0, 1))
        # ax.hist(res_dono[0], color='gray', bins=intervalos,rwidth=0.85,density=True)
        # ax.set_title('Histograma pseudo residuales normales donante', loc = "center", fontdict = {'fontsize':11, 'fontweight':'normal', 'color':'black'})
        
        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'Donante_N_condicion_'+str(cond[l])+'_trayectoria_'+str(trayec[j])+'.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()
        
        # """
        # ## Pseudo Residuales -- Aceptora Uniforme
        # """
        
        # res_acep=gaussian_HMM_pseudo_residuals(x=X[:,0], delta=delta_, gamma=gamma_, mu=lamb_[:,0], sigma=std_matrix[:,0], m=n_states)
            
        # #para obtener los graficos de la normal, descomentar las primeras tres lineas
        # fig, ax = plt.subplots()
        # plt.axhline(y=1)
        # ax.hist(x=res_acep[1], color='gray', rwidth=0.5,density=True)
        # ax.set_title('Histograma pseudo residuales uniforme aceptora', loc = "center", fontdict = {'fontsize':11, 'fontweight':'normal', 'color':'black'})
        
        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'Aceptora_U_condicion_'+str(cond[l])+'_trayectoria_'+str(trayec[j])+'.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()
        
        
        # """
        # ## Pseudo Residuales -- Donante uniforme
        # """
        
        # res_dono=gaussian_HMM_pseudo_residuals(x=X[:,1], delta=delta_, gamma=gamma_, mu=lamb_[:,1], sigma=std_matrix[:,1], m=n_states)
            
        # #para obtener los graficos de la normal, descomentar las primeras tres lineas
        # fig, ax = plt.subplots()
        # plt.axhline(y=1)
        # ax.hist(x=res_dono[1], color='gray', rwidth=0.5,density=True)
        # ax.set_title('Histograma pseudo residuales uniforme donante', loc = "center", fontdict = {'fontsize':11, 'fontweight':'normal', 'color':'black'})
        
        # # Guardar la figura en formato PDF en el directorio especificado
        # ruta_destino = os.path.join(directorio_destino, 'Donante_U_condicion_'+str(cond[l])+'_trayectoria_'+str(trayec[j])+'.pdf')
        # plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
        # plt.show()