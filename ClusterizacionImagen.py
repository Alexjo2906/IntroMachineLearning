"""
Created on Sun Jun 10 22:58:24 2018

@author: Alejandro Germosen
"""

# Declaración de librerias
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

# Importación de data
imagen = Image.open('Bandera.png')
bandera = np.asarray(imagen)

# Declaracion de colores
red = bandera[:224,:225,[0]]
green = bandera[:224,:225,[1]]
blue = bandera[:224,:225,[2]]

X = np.array(list(zip(red[0], green[0], blue[0]))) # organizar colores

k = 10 # Cantidad de clusters (colores)
posRuido = 20

bandera = bandera.reshape((50400, 3))

from copy import deepcopy

# Distancia Euclidiana
def dist(a, b, ax = 1):
    return np.linalg.norm(a - b, axis=ax) # sqrt((a0^2-b0^2)+(an^2-bn^2))

correr = 0
seed = 100
cantidadRun = 20

for correr in range(0, cantidadRun + 1):
    randSeed = np.random.seed(seed)
    
    # Inicializar centroides
    C_x = np.random.randint(0, np.max(red)-posRuido, size=k)
    C_y = np.random.randint(0, np.max(green)-posRuido, size=k)
    C_z = np.random.randint(0, np.max(blue)-posRuido, size=k)
    C = np.array(list(zip(C_x, C_y , C_z)), dtype=np.int32) #uint8 ideal
    print(C)
    print correr
    print seed
    
    C_old = np.zeros(C.shape) # Guardar espacio para centroides anteriores.
    clusters = np.zeros(len(bandera)) # Guardar espacio para no. de cluster.
    error = dist(C, C_old, None) # Distancia entre centroides viejos y nuevos.

    while error >= 1: # Se correrá el loop hasta q error sea menor a uno.
        vacio = False
        # Asignar cada pixel al centroide mas cercano.
        for i in range(len(bandera)): # i=0 hasta i=len(X)=224
            distances = dist(bandera[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Guardar los centroides anteriores.
        C_old = deepcopy(C)
        # Calcular los nuevos centroides con el average por cluster.
        for i in range(k):
            points = [bandera[j] for j in range(len(bandera)) if clusters[j] == i]
            # Las siguientes lineas se usan para corregir
            # cuando hay un cluster vacio.
            if points == 0:
                vacio = True
            if vacio:
                break
            C[i] = np.mean(points, axis=0) # calcular nuevos centroides
        
        error = dist(C, C_old, None)
        print error
        # Las siguientes lineas se usan para corregir
        # cuando hay un cluster vacio.
        if math.isnan(error):
            error = 0
            vacio = True
            break
      
    if (vacio == True and error == 0):
        correr = correr + 1
        seed = seed + 1
    else:
        break

C = C.astype(int)
print C # Imprimir los centroides resultantes

# Impresión de imagen. 
bandera.setflags(write = 1)
for ii in range(len(bandera)): # i=0 hasta i=len(X)=224
        distances = dist(bandera[ii], C)
        cluster = np.argmin(distances)
        #clusters[i] = cluster
        for jj in range(k-1):
            if cluster == jj:
                bandera[ii] = C[jj]

bandera = bandera.reshape((224, 225, 3))
plt.imshow(bandera)
plt.show    