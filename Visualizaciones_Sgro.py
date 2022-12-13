#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# # Bengaluru, Capital de Karnataka, India. Es una ciudad que cuenta con más de 12.000 restaurantes. Zomato es una app para elegir el mejor restaurante para ir a comer. 
# ## Por eso, en un entorno tan competitivo para los restaurantes, haremos una análisis de las variables más importantes del dataset, que la aplicación ha obtenido con el tiempo.
# ### Las variables más importantes son:
# + ***name:*** Nombre del restaurante, importante para reconocer cada uno de ellos.
# + ***rate:*** Esta variable resulta interesante porque nos permite conocer rápidamente los restaurantes mejor puntuados. 
# + ***votes:*** El número de votos es importante ya que también permite indicar si existe alguna relación entre los restaurantes más votados y su puntuación.
# + ***cuisines:*** La variable del tipo de cocina, puede indicarnos algún patrón por parte de los consumidores y la tendencia sobre qué es más probable que las personas de Bengaluru quieran comer. 
# + ***approx_cost:*** Este valor, sobre el costo aproximado cada 2 personas es relevante para conocer el nivel de costo promedio en el que incurren los consumidores y a su vez si existe alguna relación entre la satisfacción y el costo de la comida que se pidió.

# In[2]:


df = pd.read_csv('zomato.csv', sep = ',') #cargamos el dataset.


# In[3]:


df.head() #una primera impresión.


# In[4]:


df.describe() #funciones exploratorias.


# In[5]:


df.info()


# In[6]:


df.shape


# ##### Empecemos con los gráficos de matplotlib para entender mejor la situación.

# In[7]:


x = df['rate'] # vamos a tratar de encontrar una relación precio-calidad.
y = df['approx_cost(for two people)']


# In[8]:


df['rate'] = df['rate'].replace(regex=r'/.*', value='') #necesitamos quitarle /5 al rate.


# In[9]:


df['rate'] = df['rate'].replace(regex=' ', value='') # #necesitamos ahora sacamos posibles espacios que existan luego del /5.


# In[10]:


df['rate'].astype(str) #lo convertimos a str porque lo está tomando como float y no lo grafica matplotlib.


# In[11]:


def rate_bien(valor):
    if(valor=="NEW" or valor=="nan" or valor=="-"):
        return np.nan
    return float(str(valor).replace("/5","")) #esta funcion "rateclean" báscicamente al aplicarsela a la columna rate, nos permite eliminar un str "NEW", los nan y valores "-".


# In[12]:


df["rate"]=df["rate"].apply(rate_bien) #aplicamos la función.


# In[13]:


df["rate"].isnull().sum() #vemos si hay nulls.


# In[14]:


df["rate"]=df["rate"].fillna(df["rate"].mean()) #llenamos los nulls con promedio en general de los rate.


# In[15]:


df["rate"].isnull().sum() #no hay más nulls.


# In[16]:


df['rate'].astype(int) #llevo la columna entero para facilitar la visualización.


# In[17]:


df["approx_cost(for two people)"]=df["approx_cost(for two people)"].apply(lambda x:str(x).replace(",","")) #quitamos las comas de los resultados de esta columna.


# In[18]:


df["approx_cost(for two people)"]=df["approx_cost(for two people)"].fillna(df["approx_cost(for two people)"].astype(float).mean()) #rellenamos los nan con promedios.


# In[19]:


u_rates = df['rate'] #Voy a intentar gráficar con valores únicos (para obtener categorías).


# In[20]:


u_rates = u_rates.unique()


# In[21]:


u_rates


# In[22]:


u_rates = pd.DataFrame(u_rates)


# In[23]:


u_rates


# In[24]:


cars = u_rates[0][0:5]

data = df['votes'][0:5]

fig,ax = plt.subplots(figsize =(10, 7)) #Vamos a graficar que nivel de puntuación es el que los consumidores han votado.
ax.pie(data, labels = cars,)
ax.legend()
plt.show()


# #### Este gráfico de torta nos demuestra que la mayor cantidad de votos están en el 3.7 / 3.8 de puntuación. Por lo tanto los restaurantes de Bengaluru son bastante buenos.

# In[25]:


plt.scatter(x = df['rate'], y = df["approx_cost(for two people)"], alpha = 0.5, color = "orange") # este scatter nos permite ver si hay alguna relación precio / calidad.
plt.show()


# #### Si bien el gráfico no es lo suficientemente legible, podemos ver que el costo de la comida es (relativamente) proporcional al número de puntuación de satisfacción en general. 

# In[26]:


x = df['rate']
counts, bins = np.histogram(x)  #con este histograma podemos constatar que es real lca cifra de 3.7 / 3.8 como la más votada. 
plt.stairs(counts, bins)
plt.hist(bins[:-1], bins, weights=counts, color = 'green')


# #### Este gráfico nos demuestra que en general las personas son más propensas a votar entre 3,7 y 4,0.  

# #### Ahora avancemos con los gráficos de seaborn

# In[27]:


plt.figure(figsize=(10,5))
sns.countplot(y="online_order", data=df, palette = 'dark') #Este gráfico nos va a permitir entender el total de pedidos online o yendo físicamente.


# #### La gran mayoría de las personas han comprado por internet.

# In[28]:


plt.figure(figsize=(10,5))
sns.countplot(y= "book_table", data=df, palette = 'dark') #con este gráfico podemos ver el total de personas que han hecho reserva en el local al que han ido.


# #### El resultado está claro, la gran mayoría no hace reserva, posiblemente porque piden mayormente por delivery, más adelante veremos si es verdad.

# In[56]:


restaurantes = df.rest_type.value_counts() #Vamos a ver que hay muchas categorías de restaurantes, que no se podrían graficar todas.


# In[57]:


restaurantes_1000= restaurantes[restaurantes<1000] #hacemos entonces una variable que nos permita obtener los restaurantes con al menos 1000 valores.


# In[58]:


restaurantes_1000


# In[59]:


def categoria_restaurante(valor):
    if valor in restaurantes_1000: #Esta función lo que hace es que al aplicarla a la columna, aquellos restaurantes con 1000 valores o menos se consideren "otros".
        return "others"
    else:
         return valor


# In[60]:


df.rest_type=df.rest_type.apply(categoria_restaurante) #aplciamos la función a la columna.
df.rest_type


# In[61]:


df['rest_type']


# In[62]:


sns.histplot(data = df, x = 'rest_type', color = 'yellow', hue='rest_type') #Grafiquemos ahora entonces las categoría de restaurantes.
plt.xticks(rotation=90)


# #### El tipo de restaurante Quick Bites, comidas rápidas, es el más cómun, seguido de cena casual. 

# In[36]:


df=df.rename(columns={"listed_in(type)":"dish_type"}) #cambiamos el nombre de la columna de tipo de alimento.


# In[37]:


plt.figure(figsize=(12,5))
sns.countplot(x = "dish_type", data = df, hue="online_order", palette = 'husl')  #aplicamos ahora para conocer el tipo de plato.


# #### Vemos entonces la relación con lo comentado recién, las personas piden online delivery mayormente. También hay un cierto número de personas que hacen "Take-away" es decir comprar en el local para llevar.

# In[63]:


ubicacion = df.location.value_counts() #Vamos a ver que hay muchas categorías de ubicaciones, que no se podrían graficar todas.


# In[64]:


ubicacion


# In[65]:


ubicacion_100= ubicacion[ubicacion<100] #hacemos entonces una variable que nos permita obtener las ubicaciones con al menos 1000 valores.


# In[66]:


ubicacion_100


# In[67]:


def categoria_ubicacion(valor):
    if valor in ubicacion_100: #Esta función lo que hace es que al aplicarla a la columna, aquellas ubicaciones con 1000 valores o menos se consideren "otras".
        return "others"
    else:
         return valor


# In[68]:


df.location=df.location.apply(categoria_ubicacion) #aplciamos la función a la columna.
df.location


# In[69]:


plt.figure(figsize=(30,10))
sns.countplot(x= "location", data=df, palette = "Set2") #Graficamos entonces la columna con countplot.
plt.xticks(rotation=90)
ax.legend()


# #### Vemos en este gráfico que BTM,HSR y Koramangala 5th Block son las ubicaciones con mayor cantidad. 

# In[70]:


plt.figure(figsize=(30,10))
sns.countplot(x= "location", data=df, hue = 'online_order', palette = 'dark') #Graficamos entonces la columna con countplot.
plt.xticks(rotation=90)
ax.legend()


# #### Con este gráfico podemos ver que hay una diferencia cuando vemos el mismo gráfico dividido en pedidos online y pedidos en el local. BTM mantiene el primer puesto en ambos, pero aparencen Whitefield como el segundo lugar con mayor pedidos en el local.

# In[71]:


cocinas = df.cuisines.value_counts() #Vamos a ver que hay muchas categorías de ubicaciones, que no se podrían graficar todas.


# In[72]:


cocinas


# In[73]:


cocinas_200= cocinas[cocinas<200] #hacemos entonces una variable que nos permita obtener las ubicaciones con al menos 1000 valores.


# In[74]:


cocinas_200


# In[75]:


def categoria_cocinas(valor):
    if valor in cocinas_200: #Esta función lo que hace es que al aplicarla a la columna, aquellas ubicaciones con 1000 valores o menos se consideren "otras".
        return "others"
    else:
         return valor


# In[76]:


df.cuisines=df.cuisines.apply(categoria_cocinas) #aplciamos la función a la columna.
df.cuisines


# In[77]:


plt.figure(figsize=(30,10))
sns.countplot(y= "cuisines", data=df, palette = "Set2") #Graficamos entonces la columna con countplot.
plt.xticks(rotation=90)
ax.legend()


# #### Ignorando "otros" podemos ver que la cocina más elegida es la del Norte de India. Es llamativo esto, ya que Bengaluru se encuentra en la región Sur del país. 
