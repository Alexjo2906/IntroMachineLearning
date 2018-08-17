# In[1]:


# Declaración de librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importación de data
data_file = pd.read_csv('winequality-white.csv')
#np.random.shuffle(data_file)
data_file = data_file[:(len(data_file) - 2)]
print data_file.shape

datafile = np.genfromtxt('winequality-white.csv', dtype=float,  delimiter=',')
datafile = datafile[1:]
datafile = np.swapaxes(datafile,0,1)

# Declaración de clases
fixed_acidity = data_file['fixed acidity']
volatile_acidity = data_file['volatile acidity']
citric_acid = data_file['citric acid']
residual_sugar = data_file['residual sugar']
chlorides = data_file['chlorides']
free_sulfur_dioxide = data_file['free sulfur dioxide']
total_sulfur_dioxide = data_file['total sulfur dioxide']
density = data_file['density']
pH = data_file['pH']
sulphates = data_file['sulphates']
alcohol = data_file['alcohol']
quality = data_file['quality']


# In[2]:


cantidadFeatures = 1

# Declaración de variables
NumeroFilas = np.size(datafile, 1)
NumeroColumnas = np.size(datafile, 0)
x = np.zeros((NumeroFilas, cantidadFeatures))
D_old = np.ones((NumeroFilas, 1))

# Extraer la data que se utilizará y darle forma
for i in range(0, cantidadFeatures):
    trainingData = datafile[i]
    trainingData = np.reshape (trainingData, (NumeroFilas, 1)) # C-like index ordering
    D_old = np.concatenate((D_old, trainingData), axis = 1)

D_new = D_old[:(NumeroFilas - 2)]
r = quality


# In[3]:


D = D_new

# funcion de aprendizaje: w = (D^T .* T)^-1 .* D^T .* r
wParte1 = np.dot((np.transpose(D)), D)
wParte2 = np.dot((np.transpose(D)), r)
w = np.dot((np.linalg.inv(wParte1)), wParte2)
print w


# In[4]:


wFlip = np.flip(w,0)

# funcion de evaluacion: ecuacion = g = w^T * x
ecuacion = np.polyval(wFlip, D[:,1])
print ecuacion
plt.figure(0)
plt.title('Regresion lineal')
plt.xlabel('Fixed Acidity')
plt.ylabel('Quality')
plt.scatter(D[:,1], r, c='black', s=10)
plt.plot(D[:,1], ecuacion, 'r-')


# In[5]:


x = D[:, 1]
w = np.reshape (w, (2, 1)) # C-like index ordering
print w
print D
g = np.dot(np.transpose(w), np.transpose(D))
print g


# In[6]:


r = np.reshape (r, (4896, 1)) 

eParte1 = np.subtract(r,g)
eParte2 = np.power(eParte1[0], 2)
err = np.divide(np.sum(eParte2,0),2)
print err


# In[7]:


power2 = np.power(D[:,1],2)
power2 = np.reshape (power2, (NumeroFilas - 2, 1))
D2 = np.concatenate((D, power2), axis = 1)
print D2


# In[8]:


w2Parte1 = np.dot((np.transpose(D2)), D2)
w2Parte2 = np.dot((np.transpose(D2)), r)
w2 = np.dot((np.linalg.inv(w2Parte1)), w2Parte2)
print w2


# In[9]:


w2Flip = np.flip(w2,0)
ecuacion2 = np.polyval(w2Flip, D2[:,1])
print ecuacion2
plt.figure(1)
plt.title('Regresion cuadratica')
plt.xlabel('Fixed Acidity')
plt.ylabel('Quality')
plt.scatter(D2[:,1], r, c='black', s=10)
plt.plot(D2[:,1], ecuacion2, 'r-')


# In[10]:


g2 = np.multiply((np.transpose(w2)), D2)
print g2
g2Sum = np.sum(g2,1)
print g2Sum


# In[11]:


e2Parte1 = np.subtract(r,g2Sum)
e2Parte2 = np.power(e2Parte1[0], 2)
err2 = np.divide(np.sum(e2Parte2,0),2)
print err2


# In[12]:


power3 = np.power(D[:,1],3)
power3 = np.reshape (power3, (NumeroFilas - 2, 1))
D3 = np.concatenate((D2, power3), axis = 1)
print D3


# In[13]:


w3Parte1 = np.dot((np.transpose(D3)), D3)
w3Parte2 = np.dot((np.transpose(D3)), r)
w3 = np.dot((np.linalg.inv(w3Parte1)), w3Parte2)
print w3


# In[14]:


w3Flip = np.flip(w3,0)
ecuacion3 = np.polyval(w3Flip, D3[:,1])
print ecuacion3
plt.figure(2)
plt.title('Regresion cubica')
plt.xlabel('Fixed Acidity')
plt.ylabel('Quality')
plt.scatter(D3[:,1], r, c='black', s=10)
plt.plot(D3[:,1], ecuacion3, 'r-')


# In[15]:


g3 = np.multiply((np.transpose(w3)), D3)
print g3
g3Sum = np.sum(g3,1)
print g3Sum


# In[16]:


e3Parte1 = np.subtract(r,g3Sum)
e3Parte2 = np.power(e3Parte1[0], 2)
err3 = np.divide(np.sum(e3Parte2,0),2)
print err3