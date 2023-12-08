
"""
----------------------------------------------------------------------------
                       Modelo umbral
----------------------------------------------------------------------------

"""


"""
----------------------------------------------------------------------------
             Datos_estocasticos_Gaussiano_3_estados
----------------------------------------------------------------------------

"""
"""
----------------------------------------------------------------------------
                       # Paquetes #
----------------------------------------------------------------------------  
"""

import numpy as np
import pandas as pd
# from scipy.stats import norm
import matplotlib.pyplot as plt
# from hmmlearn import hmm
from Definiciones import PoissonHMM
# from Definiciones import pois_HMM_pseudo_residuals
# from Definiciones import duracion_rafaga
# import os

"""
----------------------------------------------------------------------------
              # Simulacion Estocastica HMM 3 estados #
----------------------------------------------------------------------------  
"""

# n_states = 3 # Numero de estados para HMM

# # Medias para el Background
# m1 = 60 #nivel de fotones esperados del transfondo captados por el detectro de la aceptora
# m2 = 90 #nivel de fotones esperados del transfondo captados por el detectro de la donante

# ef1 = 0.27 #Eficiencia para el estado 1 FRET
# ef2 = 0.94 #Eficiencia para el estado 2 FRET

# m1a =  10.10 # media ajustada para la aceptora en el primer estado
# m1d = m1a*(1-ef1)/ef1 # media ajustada para la donante en el primer estado
#                       # se hace de esa manera para que de la eficiencia deseada

# m2a = (m1a+m1d)*ef2   # media ajustada para la donante en el segundo estado,
#                       # se hace de esta forma para m1a+m1d=m2a+m2d
# m2d = m2a*(1-ef2)/ef2 # media ajustada para la donante en el primer estado
#                       # se hace de esa manera para que de la eficiencia deseada

# # Matriz de medias, primer estado es el background, 
# # los otros dos el estado FRET mas las tasa del background
# means = np.array([[m1, m2],
#                   [m1a+m1, m1d+m2],
#                   [m2a+m1, m2d+m2]])

# #print('medias reales:',means)


# # Probabilidad inicial
# startprob = np.array([1, 0, 0])


# # Matriz de transicion 
# transmat = np.array([[0.80,0.17,0.03],
#                      [0.48,0.44,0.08],
#                      [0.33,0.45,0.22]])

# # Matriz de transicion ajustada, hay que eliminar
# # primera fila y columna que es la del background
# a=np.delete(transmat, 0,0)
# gamma_=np.delete(a, 0,1)
# for i in range(n_states-1):
#   gamma_[i,:]=gamma_[i,:]/(np.sum(gamma_,axis=1)[i]) 

  
# #print('Matriz de transicion ajustada real:',np.round(gamma_,2))

# # Construir modelo con los parametros predefinidos
# np.random.seed(1)
# gen_model = PoissonHMM(n_components=n_states,n_iter=1000, init_params="mcs", params="cmt")

# # parámetros, las medias de los componentes
# gen_model.startprob_ = startprob
# gen_model.transmat_ = transmat
# gen_model.means_ = means

# # Generar muestra
# X, Z = gen_model.sample(30000) # X es la muestra y Z el camino Viterbi


"""
----------------------------------------------------------------------------
              # Simulacion Estocastica HMM 4 estados#
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

print(np.array([[m1, m2],[m1a, m1d],[m2a, m2d],[m3a, m3d]]))

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
  
#print('Matriz de transicion ajustada real:',np.round(gamma_,2))

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
               # ### Guardar los conteos ### #
----------------------------------------------------------------------------  
"""

suma_columnas = X[:, 0] + X[:, 1]

# Agrega la tercera columna a la matriz original
Y = np.column_stack((X, suma_columnas))

# Convierte la matriz X en un DataFrame
df = pd.DataFrame(Y, columns=["Aceptora", "Donante", "Suma_Aceptora_Donante"])

# Especifica la ubicación y el nombre del archivo CSV en el que deseas guardar los datos
nombre_archivo = "Datos_Umbral.csv"

# Guarda el DataFrame en un archivo CSV
df.to_csv(nombre_archivo, index=False)

print(f"Los resultados se han guardado en {nombre_archivo}")

"""
-------------------------------------------------------------------------------
            #  Histogramas #
-------------------------------------------------------------------------------  
"""

# Supongamos que las tres columnas son Canal_Aceptora, Canal_Donante y Suma
canal_aceptora = Y[:, 0]
canal_donante = Y[:, 1]
suma_canales = Y[:, 2]

#  # Especificar la ruta completa al directorio de destino
# directorio_destino = r'C:/Users/sebas/Desktop/Enzimas/Datos_simulados/Umbral_2_estados'

# # Verificar si el directorio existe y, si no, crearlo
# if not os.path.exists(directorio_destino):
#     os.makedirs(directorio_destino)

# Crear histograma para Canal_Aceptora
plt.hist(canal_aceptora, bins=20, alpha=0.5, label='Canal Aceptora', rwidth=0.9, color='black')
plt.xlabel('Número de fotones')
plt.ylabel('Frecuencia')
# plt.legend()
plt.title('Histograma para canal aceptora')

# Guardar la figura en formato PDF en el directorio especificado
# ruta_destino = os.path.join(directorio_destino, 'his_acep_umbral_Gaussiano_acep_'+str(m1)+'_don_'+str(m2)+'.pdf')
# plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
plt.show()

# Crear histograma para Canal_Donante
plt.hist(canal_donante, bins=20, alpha=0.5, label='Canal Donante', rwidth=0.9, color='black')
plt.xlabel('Número de fotones')
plt.ylabel('Frecuencia')
# plt.legend()
plt.title('Histograma para canal donante')

# Guardar la figura en formato PDF en el directorio especificado
# ruta_destino = os.path.join(directorio_destino, 'his_don_umbral_Gaussiano_acep_'+str(m1)+'_don_'+str(m2)+'.pdf')
# plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
plt.show()

# Crear histograma para la Suma de Canal_Aceptora y Canal_Donante
plt.hist(suma_canales, bins=20, alpha=0.5, label='Suma Canales', rwidth=0.9, color='black')
plt.xlabel('Número de fotones')
plt.ylabel('Frecuencia')
# plt.legend()
plt.title('Histograma para total de fotones')

# Guardar la figura en formato PDF en el directorio especificado
# ruta_destino = os.path.join(directorio_destino, 'his_suma_umbral_Gaussiano_acep_'+str(m1)+'_don_'+str(m2)+'.pdf')
# plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
plt.show()
"""
---------------------------------------------
            grafica de tiempo
---------------------------------------------
"""

umbral = 160

t = 800  # Definir el punto de tiempo
lista = list(range(t))
datos = {'suma': Y[:, 2][:t]}
plt.plot(lista, datos['suma'], label='suma')
# plt.axhline(y=140, color='red', linestyle='--', label='Umbral = 140')
plt.axhline(y= umbral, color='red', linestyle='--', label='Umbral = '+str(umbral))   # Agrega una línea horizontal en y=70
plt.legend(loc='upper right')
plt.xlabel("Tiempo")

# Guardar la figura en formato PDF en el directorio especificado
# ruta_destino = os.path.join(directorio_destino, 'suma_umbral_Gaussiano_acep_'+str(m1)+'_don_'+str(m2)+'.pdf')
# plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
plt.show()


t = 1000  # Definir el punto de tiempo
# Establece el umbral

Y_ = Y.astype(float)
Y_[Y_[:, 2] < umbral, :] = np.nan
# t=800

fig, ax = plt.subplots()
lista = list(range(t))
datos = {'Acceptor': Y_[:t, 0], 'Donor': Y_[:t, 1]}
ax.plot(lista, datos['Acceptor'], label='Aceptora')
ax.plot(lista, datos['Donor'], label='Donante')
ax.legend(loc='upper right')
ax.set_xlabel("Tiempo")

# ruta_destino = os.path.join(directorio_destino, 'conteos_umbral_Gaussiano_acep_'+str(m1)+'_don_'+str(m2)+'.pdf')
# plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
plt.show()

"""
---------------------------------------------
             Correr codigo r
---------------------------------------------
"""



# Correr el archivo de r


"""
---------------------------------------------
             BIC
---------------------------------------------
"""

# Ruta del archivo CSV
ruta_csv = "C:/Users/sebas/Documents/Tesis_Enzimas/bic_values.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(ruta_csv)

# Obtener la columna 'BIC_Valor' como una lista en Python
scores = df['BIC_Valor'].tolist()

fig, ax = plt.subplots()
lista = [2,3,4,5,6,7,8,9]
datosbic = {'traye1':scores}
ax.plot(lista, datosbic['traye1'])
ax.set_xlabel("Num. Estados")
ax.set_ylabel("BIC")
ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed')

# Guardar la figura en formato PDF en el directorio especificado
# ruta_destino = os.path.join(directorio_destino, 'BIC_umbral_Gaussiano_acep_'+str(m1)+'_don_'+str(m2)+'.pdf')
# plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
plt.show()

porcen = list()

for k in range(len(scores)-1):
    porcen.append((scores[k]-scores[k+1])/scores[k]*100)
    
print(np.round(porcen,2))

"""
---------------------------------------------
             Viterbi
---------------------------------------------
"""
archivo_csv = "C:/Users/sebas/Documents/Tesis_Enzimas/datos_viterbi_r.csv"
states = pd.read_csv(archivo_csv)
states_ = np.array(states['x'])

t = 1000

states_ = states_[:t]

# Convertir 1 a 0 y 2 a 1
# states_ = np.where(states_ == 3, 2, np.where(states_ == 2, 1, np.where(states_ == 1, 0, states_)))

lam = list([100,150,70])
means_ = np.array(lam)
list_ = list(np.sort(lam))
lam_ = means_[states_[:t]]
  
# Organizar para que el estado cero sea el background
for i in range(n_states-1):
  lam_[lam_==list_[i]]=i
  
  
fig, ax = plt.subplots()
# ax.plot(lam_, color = 'green')
ax.plot(states_, color = 'green')
ax.set_ylabel('Estado')
ax.set_xlabel('Tiempo')
plt.yticks(range(1,4,1))
  
# Guardar la figura en formato PDF en el directorio especificado
# ruta_destino = os.path.join(directorio_destino, 'viterbi_umbral_Gaussiano_acep_'+str(m1)+'_don_'+str(m2)+'.pdf')
# plt.savefig(ruta_destino, format='pdf', bbox_inches='tight', transparent=True)
plt.show()  