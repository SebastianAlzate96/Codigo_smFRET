#set.seed(1)

library(depmixS4)

data_ <- read.csv("C:/Users/sebas/Documents/Tesis_Enzimas/Datos_Umbral.csv")
data <- data.frame(data_)
head(data)

# Cargar la biblioteca ggplot2
library(ggplot2)

umbral <- 160

# Filtrar los valores de Y que sean mayores que el umbral
data[data$Suma_Aceptora_Donante < umbral, ] <- NA

model <- depmix(response = list(Aceptora ~ 1, Donante~1), family=list(gaussian(),gaussian()), 
                nstates = 3 ,data = data)

# Ajustar el modelo a los datos
fit <- fit(model)

fit

# Resumen de los parámetros estimados
summary(fit)

#simulate(model,30000)


states_posterior_probs <- fit@posterior$state

# Especifica la ruta completa al directorio donde deseas guardar el archivo
directorio <- "C:/Users/sebas/Documents/Tesis_Enzimas"

# Combinar la ruta del directorio con el nombre del archivo
archivo <- file.path(directorio, "datos_viterbi_r.csv")

# Escribir los datos en un archivo CSV
write.csv(states_posterior_probs, archivo, row.names = FALSE)

lista_bic <- list()

for (i in 2:9) {
  model_ <- depmix(response = list(Aceptora ~ 1, Donante~1), family=list(gaussian(),gaussian()), 
                   nstates = i ,data = data)
  fit_ <- fit(model_)
  print(fit_)
  lista_bic[[i]] <- BIC(fit_)
}

# Convertir la lista a un dataframe
df_bic <- data.frame(Numero_de_Estados = 2:9, BIC_Valor = unlist(lista_bic))

# Ruta completa del archivo CSV
ruta_csv <- file.path(directorio, "bic_values.csv")

# Guardar el dataframe en un archivo CSV en el directorio especificado
write.csv(df_bic, file = ruta_csv, row.names = FALSE)
