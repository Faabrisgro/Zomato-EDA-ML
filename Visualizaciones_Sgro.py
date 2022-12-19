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


df = pd.read_csv('zomato.csv', sep = ',')


# In[3]:


df.head()


# In[4]:


df.describe() 


# In[5]:


df.info()


# In[6]:


df.shape


# ##### Empecemos con los gráficos de matplotlib para entender mejor la situación.

# > vamos a tratar de encontrar una relación precio-calidad.

# In[7]:


x = df['rate'] 
y = df['approx_cost(for two people)']


# In[8]:


df['rate'] = df['rate'].replace(regex=r'/.*', value='') 


# In[9]:


df['rate'] = df['rate'].replace(regex=' ', value='') 


# In[10]:


df['rate'].astype(str)


# > esta funcion "rateclean" básicamente al aplicársela a la columna rate, nos permite eliminar un str "NEW", los nan y valores "-".

# In[11]:


def rate_bien(valor):
    if(valor=="NEW" or valor=="nan" or valor=="-"):
        return np.nan
    return float(str(valor).replace("/5","")) 


# In[12]:


df["rate"]=df["rate"].apply(rate_bien) 


# In[13]:


df["rate"].isnull().sum()


# In[14]:


df["rate"]=df["rate"].fillna(df["rate"].mean())


# In[15]:


df["rate"].isnull().sum()


# In[16]:


df['rate'] = df['rate'].astype(int)


# In[17]:


df["approx_cost(for two people)"]=df["approx_cost(for two people)"].apply(lambda x:str(x).replace(",","")) 


# In[18]:


df["approx_cost(for two people)"]=df["approx_cost(for two people)"].fillna(df["approx_cost(for two people)"].astype(float).mean()) 


# > cambiamos el nombre de la columna de tipo de local. 

# In[19]:


df=df.rename(columns={"listed_in(type)":"local_type"})


# > Debemos ajustar el tipo de restaurante para graficarlo

# In[20]:


restaurantes = df.rest_type.value_counts() 
restaurantes_1000= restaurantes[restaurantes<1000]
restaurantes_1000


# In[21]:


def categoria_restaurante(valor):
    if valor in restaurantes_1000:
        return "others"
    else:
         return valor


# In[22]:


df.rest_type=df.rest_type.apply(categoria_restaurante) 
df.rest_type


# In[23]:


df['rest_type']


# > Vamos a hacer el primer gráfico con estos cambios

# In[24]:


fig, ax = plt.subplots()

x = df['rate'][0:25000]

y = df['votes'][0:25000]
 

plt.bar(x,y)

plt.xticks(x,x)
ax.set_xlabel('Rango de votación n/5')
ax.set_ylabel('Cantidad de votos')
ax.set_title('Rango de votación por cantidad de votos', style = 'italic')
plt.show()


# #### Este gráfico de torta nos demuestra que la mayor cantidad de votos están en el rango 4/5 de puntuación. Por lo tanto los restaurantes de Bengaluru son bastante buenosy su servicio satisfacen mayormente las expectativas de los clientes.

# In[25]:


df['approx_cost(for two people)'].fillna(0)
df['approx_cost(for two people)']= df['approx_cost(for two people)'].astype(float)
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].fillna(0)


# In[26]:


x= df['local_type']
y_df= df.groupby(by=["local_type"]).mean()
y_df= y_df.drop(['rate'], axis=1)
y_df= y_df.drop(['votes'], axis=1)
y_df


# >Luego de limpiar la columna y dejarla lista para el ploteo, procedemos a ver el resultado del costo aproximado por tipo de plato

# In[27]:


plt.style.use('seaborn')
plt.figure(figsize=(16,8))
plt.plot(y_df)
plt.ylabel('Costo aproximado')
plt.xlabel ('Tipo de local')
plt.title('Costo promedio para dos personas por tipo de restaurante',style= 'italic')
plt.show()



# #### Este gráfico nos da a entender rápidamente que tipo de restaurante es el que tiene mayor costo promedio, en este caso podemos ver que los restaurantes o locales relacionados con bebidas y vida nocturan son los más elevados, seguido de Pubs y Bars. 

# > Ahora avancemos con los gráficos de seaborn
# 
# > Este gráfico nos va a permitir entender el total de pedidos online o yendo físicamente.

# In[28]:


plt.style.use('seaborn')
plt.figure(figsize=(10,5))
sns.countplot(y="online_order", data=df, palette = 'dark').set(title = 'Cantidad de pedidos online y físicos')
plt.xlabel('Número de pedidos')
plt.ylabel('¿El pedido fue online?')


# #### Con el resultado del gráfico vemos que se pide mayormente comida online. 

# In[29]:


plt.style.use('seaborn')
plt.figure(figsize=(10,5))
sns.countplot(y= "book_table", data=df, palette = 'dark').set(title = 'Cantidad de pedidos con reserva y sin reserva')
plt.xlabel('Número de pedidos')
plt.ylabel('¿Realizó reserva?')


# #### El resultado está claro, la gran mayoría no hace reserva, posiblemente porque piden mayormente por delivery, más adelante veremos si es verdad.

# In[46]:


plt.style.use('fast')
sns.histplot(data = df, x = 'rest_type', hue='rest_type').set(title = 'Cantidad de pedidos por tipo de restaurante')
plt.ylabel('Número de pedidos')
plt.xlabel('Tipo de restaurante')
plt.xticks(rotation=90)


# #### El tipo de restaurante Quick Bites, comidas rápidas, es el más cómun, seguido de cena casual. 

# In[31]:


plt.figure(figsize=(12,5))
sns.countplot(x = "local_type", data = df, hue="online_order", palette = 'husl').set(title = 'Cantidad de pedidos con reserva y sin reserva')
plt.xlabel('Tipo de local')
plt.ylabel('¿Número de pedidos') 


# #### Vemos entonces la relación con lo comentado recién, las personas piden online delivery mayormente. También hay un cierto número de personas que hacen "Take-away" es decir comprar en el local para llevar.

# In[32]:


ubicacion = df.location.value_counts() 


# In[33]:


ubicacion


# In[34]:


ubicacion_100= ubicacion[ubicacion<100] 


# In[35]:


ubicacion_100


# In[36]:


def categoria_ubicacion(valor):
    if valor in ubicacion_100: 
        return "others"
    else:
         return valor


# In[37]:


df.location=df.location.apply(categoria_ubicacion) 
df.location


# In[38]:


plt.figure(figsize=(16,8))
chart= sns.barplot( x="location", y='votes', data=df, hue = "online_order")
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
chart.set(title = 'Cantidad de pedidos en línea y físicos por ubicación') 
plt.xlabel('Cocina')
plt.ylabel('Cantidad de votos') 


# #### Con este gráfico podemos ver que Lavelle Road es la ubicación con más pedidos online y a su vez, Koramangala 5th Block con más pedidos físicos. 

# In[39]:


cocinas = df.cuisines.value_counts() 
cocinas


# In[40]:


cocinas_200= cocinas[cocinas<100] 


# In[41]:


cocinas_200


# In[42]:


def categoria_cocinas(valor):
    if valor in cocinas_200:
        return "others"
    else:
         return valor


# In[43]:


df.cuisines=df.cuisines.apply(categoria_cocinas) 
df.cuisines


# In[44]:


plt.figure(figsize=(16,8))
chart= sns.barplot( x="cuisines", y='votes', data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
chart.set(title = 'Cantidad de votos por tipo de cocina') 
plt.xlabel('Tipos de Cocinas')
plt.ylabel('Cantidad de votos') 


# #### Ignorando "otros" podemos ver que la cocina más elegida es la de la cocina "Chinese,Thai, Momos". Es decir cocina china, tailendesa y momos. Además muy cerquita se encuentra la cocina del Norte de India y Mughlai. Al ser Bengaluru una región del sur de India llama la atención la presencia de cocina del norte. 

# In[45]:


plt.figure(figsize=(16,8))
chart= sns.barplot( x="location", y='votes', data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
chart.set(title = 'Cantidad de votos por ubicación') 
plt.xlabel('Ubicaciones')
plt.ylabel('Cantidad de votos')


# #### Con este gráfico podemos medir que ubicaciones han realizado mayor cantidad de votaciones y en este caso fue Church Street. Este dato puede ser útil para saber que zonas tienen de a hacer más votaciones y con ello aumentar el número de respuestas en encuestas o valoración del servicio. Lo que podría ayudar a los restaurantes a mejorar sus servicios basado en las puntuaciones que reciben.
