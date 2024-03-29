"""
----------------------------------------------------------------------------
             Datos Experimentales 3 estados
----------------------------------------------------------------------------

"""



"""
----------------------------------------------------------------------------
             POrcentages de disminucion
----------------------------------------------------------------------------

"""

"""
----------------------------------------------------------------------------
                       # Paquetes #
----------------------------------------------------------------------------  
"""


import numpy as np
import pandas as pd
from hmmlearn import hmm

cond=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0]
trayec=[1,2,3,4]

table_Decrease = np.zeros((1,9))
table_porcentages = np.zeros((1,8))
for n in range(len(cond)):
    np.random.seed(1)
    filename='C:/Users/Profesor/OneDrive - Estudiantes ITCR/Escritorio/Enzimas/datos/data_all.csv'
    df=pd.read_csv(filename,sep=",")
    new_column_names = ['num', 'Time', 'Acceptor', 'donor', 'condition', 'trayectory']
    df.columns = pd.Index(new_column_names)



    df=df[df['condition']==cond[n]]
    df=df[df['trayectory']==trayec[3]]


    X=np.zeros([len(df['Acceptor']),2])
    X[:,0] = df['Acceptor']
    X[:,1] = df['donor']


    scores=list()

    for n_comp in range(2,11):
        model = hmm.GaussianHMM(n_components=n_comp,n_iter=1000,covariance_type="full",random_state=1)
        model.fit(X)
        c = n_comp*n_comp+2*n_comp-1
        scores.append(c*np.log(len(X[:,0]))-2*model.score(X))

    porcen = list()

    for k in range(len(scores)-1):
        porcen.append((scores[k]-scores[k+1])/scores[k]*100)

    table_Decrease = np.concatenate((table_Decrease,np.array([scores])),0)
    table_porcentages = np.concatenate((table_porcentages,np.array([porcen])),0)

table_Decrease = np.delete(table_Decrease, 0,0) # eliminar filas de ceros
table_porcentages = np.delete(table_porcentages, 0,0)

#row_indices = ["Background", "State 1", "State 2", "State 1 Adjusted"]
#column_names = ["2 states", "3 states", "4 states", "5 states", "6 states", "7 states", "8 states", "9 states", "10 states"]
#table_Decrease_ = pd.DataFrame(np.round(table_Decrease,2), columns=column_names)
#table_Decrease_.to_csv('table_Decrease.csv')
#print(table_Decrease_)

#row_indices = ["Background", "State 1", "State 2", "State 1 Adjusted"]
column_names = ["2-3 decrease", "3-4 decrease", "4-5 decrease", "5-6 decrease",
                   "6-7 decrease", "7-8 decrease", "8-9 decrease", "9-10 decrease"]
table_porcentages_ = pd.DataFrame(np.round(table_porcentages,2), columns=column_names)
# table_porcentages_.to_csv('tabla_porcentajes_BIC_trayectoria_4.csv')
print(table_porcentages_)