#!/usr/bin/env python
# coding: utf-8

# # **Proyecto Data Science Parte III y IV**

# ## Introducción 

# **Bengaluru, Capital de Karnataka, India**. Es una ciudad que cuenta con más de 12.000 restaurantes. **Zomato** es una app para elegir el mejor restaurante para ir a comer.  
# 
# Esta ciudad es una de las príncipales capitales de desarrollo tecnológico de la India. Por lo que las personas son propensas a consumir por delivery. También en la India existen ciertas culturas con restricciones alimenticias (vegetarianos) por lo que analizar la información de este dataset puede ser de ayuda para nuevos emprendedores que quieran abrir un local en esta región, para que puedan decidir más seguros, sobre en que parte de la ciudad, qué tipo de restaurant les conviene y maximizar sus chances y ventajas comepetitivas.

# ## Definición del objetivo

# + ¿Cómo podemos determinar qué tipo de cocina y en que zona tiene más probabilidad un local gastronómico de ser exitoso? 
# 
# + ¿Hay más posibilidades de tener éxito ofreciendo pedidos de comida online? ¿Y si no, cómo afectaría al negocio?
# 
# + ¿Cómo afecta la puntuación del restaurante (del 1 al 5) en la decisión de los consumidores de comprar comida?

# ## Contexto Comercial

# El dataset con el que contamos para realizar el análisis concluyente y responder las preguntas objetivos, contiene información sobre el nombre, url, dirección del restaurante, si tiene pedido online, si se hizo reserva, la puntuación, la cantidad de votos, datos de contacto, ubicación del restaurante, el tipo de restaurante, las reviews, el tipo de cocina, el costo aproximado por dos personas. 
# 
# Con estos datos se comenzará a hacer un análisis exploratorio que nos permita determinar y concluir qué es lo mejor para un nuevo emprendedor en esta región de la India que busca poner un local gastronómico.

# ## Problema comercial

# + No sabemos como está distribuido la puntuación y los votos. ¿Qué es puntuación es más común entre los restaurantes de Bengaluru? 
# + Y sobre el costo aproximado, nos gustaría saber que tipo de cocina tiene costos más elevados. 
# + ¿Hay más pedidos online o la gente va más a los locales?
# + Y en cuanto a reservas, ¿las personas reservan o simplemente van?
# + ¿Qué tipo de restaurante tiene más pedidos?
# + ¿Y qué tipo de restaurantes reciben más pedidos online?
# + ¿Podemos saber que ubicaciones piden más online y cuales reciben más visitas físicas?
# + ¿Qué tipo de cocina es más probable que los consumidores deseen comer?
# + ¿Qué ubicaciones reciben más pedidos?

# ## Contexto Analítico 
# El dataset que vamos a estar trabajando, es un .csv separado por comas. El mismo tiene 51.717 filas y 17 columnas. 
# 

# **Las variables más importantes son:**
# + ***name:*** Nombre del restaurante, importante para reconocer cada uno de ellos.
# + ***rate:*** Esta variable resulta interesante porque nos permite conocer rápidamente los restaurantes mejor puntuados. 
# + ***votes:*** El número de votos es importante ya que también permite indicar si existe alguna relación entre los restaurantes más votados y su puntuación.
# + ***cuisines:*** La variable del tipo de cocina, puede indicarnos algún patrón por parte de los consumidores y la tendencia sobre qué es más probable que las personas de Bengaluru quieran comer. 
# + ***approx_cost:*** Este valor, sobre el costo aproximado cada 2 personas es relevante para conocer el nivel de costo promedio en el que incurren los consumidores y a su vez si existe alguna relación entre la satisfacción y el costo de la comida que se pidió.

# **También nos gustaría poder implementar machine learning y obtener:**
# 
# + ***Tipos de comida*** con mayor posibilidad de ser consumidos por los clientes.
# + ***Ubicaciones*** de la ciudad con mayor número de personas que piden cierto tipo de comida.
# + ***Variables*** que pueden afectar en la decisión de un cliente si dejar un voto positivo (3-4-5) o uno negativo (0-1-2).
# + ***Posibles costos*** que se podrían esperar para cada tipo de restaurante o local.
# 
# La determinación de esta información puede ser valiosa para locales que quieran comenzar a operar en la ciudad y necesiten de estudios de mercado como el que este lo proveería.

# ## Análisis exploratorio de Datos 
# Con estas preguntas podemos empezar a trabajar en el dataset y lograr darles una respuesta.

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('zomato.csv', sep = ',') #leemos el dataset
df2 = df


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe() 


# In[6]:


df.info()


# In[7]:


df.duplicated().sum()


# In[8]:


df.duplicated().sum()
df.drop_duplicates(inplace=True)


# > dropeo de columnas innecesarias

# In[9]:


df.drop(['url','phone','dish_liked'],axis=1,inplace=True)


# > Dropeo de nulls

# In[10]:


df.dropna(how='any',inplace=True)


# > ¿Cuál es son los restaurantes con más pedidos?

# In[11]:


nombres = df['name'].value_counts() #contamos las veces que aparecen los mismos resaurantes
nombres_df = pd.DataFrame(nombres)


# In[12]:


top_10_rests = nombres_df.name.nlargest(n=10) #para graficar vamos a usar los 10 mejores
top10df= pd.DataFrame(top_10_rests)
top10df


# In[13]:


fig, ax = plt.subplots()
plt.bar(top10df.index,top10df.name, color = 'orange')
plt.xticks(rotation = 75)
ax.set_xlabel('Restaurantes')
ax.set_ylabel('Cantidad de pedidos')
ax.set_title('Top 10 restaurantes con más pedidos', style = 'italic')
plt.show()


# Aquí podemos ver los 10 restaurantes con más pedidos. Es interesante ver que el que restaurante con mayor número de pedidos es un Café. 

# > **Un poco de manipulación de datos**

# In[14]:


x = df['rate'] 
y = df['approx_cost(for two people)']


# In[15]:


df['rate'] = df['rate'].replace(regex=r'/.*', value='') 


# In[16]:


df['rate'] = df['rate'].replace(regex=' ', value='') 


# In[17]:


df['rate'].astype(str)


# > esta funcion "ratebien" básicamente al aplicársela a la columna rate, nos permite eliminar un str "NEW", los nan y valores "-".

# In[18]:


def rate_bien(valor):
    if(valor=="NEW" or valor=="nan" or valor=="-"): #elimanamos los nan, un NEW y quitamos /5
        return np.nan
    return float(str(valor).replace("/5","")) 


# In[19]:


df["rate"]=df["rate"].apply(rate_bien) 


# In[20]:


df["rate"].isnull().sum()


# In[21]:


df["rate"]=df["rate"].fillna(0)


# In[22]:


df["rate"].isnull().sum()


# In[23]:


df["approx_cost(for two people)"]=df["approx_cost(for two people)"].apply(lambda x:str(x).replace(",","")) 


# In[24]:


df["approx_cost(for two people)"]=df["approx_cost(for two people)"].fillna(df["approx_cost(for two people)"].astype(float).mean()) 


# In[ ]:





# > cambiamos el nombre de la columna de tipo de local. 

# In[25]:


df=df.rename(columns={"listed_in(type)":"local_type"})
df=df.rename(columns={"approx_cost(for two people)":"cost_per_two"})


# In[26]:


df['cost_per_two']=df['cost_per_two'].astype(str)
df['cost_per_two'] = df['cost_per_two'].apply(lambda x: x.replace(',','.')) #Usamos una función lambda para reemplazar las ',' de esta columna
df['cost_per_two'] = df['cost_per_two'].astype(float)


# In[27]:


df.name = df.name.apply(lambda x:x.title())
df.online_order.replace(('Yes','No'),(True, False),inplace=True)
df.book_table.replace(('Yes','No'),(True, False),inplace=True)
df


# > Debemos ajustar el tipo de restaurante para graficarlo

# In[28]:


restaurantes = df.rest_type.value_counts() 
restaurantes_1000= restaurantes[restaurantes<1000]
restaurantes_1000


# In[29]:


def categoria_restaurante(valor):
    if valor in restaurantes_1000:
        return "others"
    else:
         return valor


# In[30]:


df.rest_type=df.rest_type.apply(categoria_restaurante) 
df.rest_type


# In[31]:


df['rest_type']


# In[32]:


df.rate = df.rate.astype(int)
print(df.rate.unique())


# In[33]:


dfrates = df.rate.value_counts().sort_values(ascending=False)
dfrates = pd.DataFrame(dfrates)
dfrates


# In[34]:


dfrates = dfrates.rename(index={4:5, 3:4, 2:3, 1:2, 0:1})


# > Vamos a ver la distribución de votos y puntuación

# In[35]:


fig, ax = plt.subplots()

x = dfrates.index

y = dfrates.rate

plt.bar(x,y)

plt.xticks(x)
ax.set_xlabel('Rango de votación n/5')
ax.set_ylabel('Cantidad de votos')
ax.set_title('Rango de votación por cantidad de votos', style = 'italic')
plt.show()


# Este gráfico de barras nos demuestra que la mayor cantidad de votos están en el rango 4/5 de puntuación. Por lo tanto los restaurantes de Bengaluru son bastante buenos y su servicio satisfacen mayormente las expectativas de los clientes.

# In[36]:


df['cost_per_two'].fillna(0)
df['cost_per_two']= df['cost_per_two'].astype(float)
df['cost_per_two'] = df['cost_per_two'].fillna(0)


# In[37]:


y_df= df.groupby(by=["local_type"]).mean('cost_per_two')
y_df= y_df.drop(['rate'], axis=1)
y_df= y_df.drop(['votes'], axis=1)
y_df = y_df.drop(['online_order'], axis =1)
y_df = y_df.drop(['book_table'], axis = 1)
y_df


# >Luego de limpiar la columna y dejarla lista para el ploteo, procedemos a ver el resultado del costo aproximado por tipo de plato

# In[38]:


plt.figure(figsize=(10,4))
plt.plot(y_df, linewidth=5)
plt.ylabel('Costo aproximado')
plt.xlabel ('Tipo de local')
plt.title('Costo promedio para dos personas por tipo de restaurante',style= 'italic')
plt.show()


# Este gráfico nos da a entender rápidamente que tipo de restaurante es el que tiene mayor costo promedio, en este caso podemos ver que los restaurantes o locales relacionados con bebidas y vida nocturan son los más elevados, seguido de Pubs y Bars. 

# > Ahora busquemos qué restaurantes tienen el costo promedio más alto, veamos si paso lo mismo que en el gráfico de arriba

# In[39]:


rest_cost= df.groupby(by=["name"]).mean('cost_per_two')
rest_cost= rest_cost.drop(['rate'], axis=1)
rest_cost= rest_cost.drop(['votes'], axis=1)
rest_cost= rest_cost.sort_values(by = ['cost_per_two'], ascending = False)


# In[40]:


rest_cost = rest_cost.reset_index()


# In[41]:


rest_cost.columns


# In[42]:


rest_cost.loc[18,"name"] = 'Cafe - Shangri-La Hotel'


# In[43]:


fig, ax = plt.subplots()
plt.bar(rest_cost.name[0:20], rest_cost.cost_per_two[0:20], color = 'lightblue')
plt.xticks(rotation = 90)
ax.set_xlabel('Restaurantes')
ax.set_ylabel('Costo por 2 personas aproximado')
ax.set_title('Los 20 restaurantes con más costo promedio para dos personas', style = 'italic')
plt.show()


# Le Cirque Signatura - The Leela Palace es el restaurante con el costo por 2 personas aproximado, más elevado de todos los restaurantes. 

# In[44]:


bins = [(df["cost_per_two"] >= -.5) & (df["cost_per_two"] <= .5)]


# In[45]:


fig=sns.histplot(data = df, x='cost_per_two', bins = 10, kde = True).set(title = 'Distribución del costo aprox. para 2 personas')
plt.xlabel('Costo aproximado cada 2 personas')
plt.ylabel('Total de veces que se repite el valor')
plt.show()


# Lo que podemos ver con este gráfico es que claramente la mayor cantidad de restaurantes y clientes, compran comida de entre 0 a 500 usd. 

# > Ahora avancemos con los gráficos de seaborn, este gráfico nos va a permitir entender el total de pedidos online o yendo físicamente.

# In[46]:


plt.figure(figsize=(10,5))
sns.countplot(y="online_order", data=df, palette = 'dark').set(title = 'Cantidad de pedidos online y físicos')
plt.xlabel('Número de pedidos')
plt.ylabel('¿El pedido fue online?')


# Con el resultado del gráfico vemos que se pide mayormente comida online. 

# In[47]:


plt.figure(figsize=(10,5))
sns.countplot(y= "book_table", data=df, palette = 'dark').set(title = 'Cantidad de pedidos con reserva y sin reserva')
plt.xlabel('Número de pedidos')
plt.ylabel('¿Realizó reserva?')


# El resultado está claro, la gran mayoría no hace reserva, posiblemente porque piden mayormente por delivery, más adelante veremos si es verdad.

# ¿Es más caro en general, pedir online o salir a comer?

# In[48]:


fig = sns.boxplot(df,x='online_order',y='cost_per_two', hue='online_order').set(title = 'Costo aproximado para 2 personas por tipo de pedido')
plt.xlabel('¿Pedido online?')
plt.ylabel('Costo aproximado para 2 personas')
plt.show()


# La respuesta es, que es más barato en general, pedir comida online. O puede que cuando la gente sale, tiende a gastar más dinero que lo habitual si hiciera el pedido online.

# In[49]:


dfrest =  df.rest_type.value_counts().nlargest(10)
dfrest = pd.DataFrame(dfrest)
dfrest


# In[50]:


sns.barplot(x= dfrest.index, y= dfrest.rest_type).set(title = 'Cantidad de pedidos por tipo de restaurante')
plt.ylabel('Número de pedidos')
plt.xlabel('Tipo de restaurante')
plt.xticks(rotation=90)


# El tipo de restaurante Quick Bites, comidas rápidas, es el más cómun, seguido de cena casual. 

# In[51]:


plt.figure(figsize=(12,5))
sns.countplot(x = "local_type", data = df, hue="online_order", palette = 'husl').set(title = 'Cantidad de pedidos con reserva y sin reserva')
plt.xlabel('Tipo de local')
plt.ylabel('Número de pedidos') 


# Vemos entonces la relación con lo comentado recién, las personas piden online delivery mayormente. También hay un cierto número de personas que hacen "Take-away" es decir comprar en el local para llevar.

# In[52]:


ubicacion = df.location.value_counts() 


# In[53]:


ubicacion


# In[54]:


ubicacion_100= ubicacion[ubicacion<100] 


# In[55]:


ubicacion_100


# In[56]:


def categoria_ubicacion(valor):
    if valor in ubicacion_100: 
        return "others"
    else:
         return valor


# In[57]:


df.location=df.location.apply(categoria_ubicacion) 
df.location


# In[58]:


x1 = df.location.value_counts().nlargest(15)


# In[59]:


x1df = pd.DataFrame(x1)
x1df


# In[60]:


plt.figure(figsize=(16,8))
chart= sns.barplot( x="location", y='votes', data=df, hue = "online_order")
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
chart.set(title = 'Cantidad de pedidos en línea y físicos por ubicación') 
plt.xlabel('Cocina')
plt.ylabel('Cantidad de votos') 


# Con este gráfico podemos ver que Lavelle Road es la ubicación con más pedidos online y a su vez, Koramangala 5th Block con más pedidos físicos. 

# In[61]:


top_cinco_cocinas= df2.cuisines.value_counts().nlargest(n=20)
top_cinco_cocinas


# In[62]:


graph1 = sns.lineplot(top_cinco_cocinas, linewidth=5)
graph1.set(title = 'Número de tipo de cocinas')
plt.xticks(rotation=90)
plt.xlabel('Tipos de Cocinas')
plt.ylabel('Número de cocinas') 


# Este gráfico nos arroja las cocinas más comunes de Bengaluru, podemos ver que la más común de todas es la del Norte de India y China. 

# In[63]:


cocinas = df.cuisines.value_counts() 
cocinas


# In[64]:


cocinas_200= cocinas[cocinas<100] 


# In[65]:


cocinas_200


# In[66]:


def categoria_cocinas(valor):
    if valor in cocinas_200:
        return "others"
    else:
         return valor


# In[67]:


df.cuisines=df.cuisines.apply(categoria_cocinas) 
df.cuisines


# In[68]:


plt.figure(figsize=(16,8))
chart= sns.barplot( x="cuisines", y='votes', data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
chart.set(title = 'Cantidad de votos por tipo de cocina') 
plt.xlabel('Tipos de Cocinas')
plt.ylabel('Cantidad de votos') 


# Podemos ver que la cocina más votada es la de la cocina "Chinese,Thai, Momos". Es decir cocina china, tailendesa y momos. Además muy cerquita se encuentra la cocina del Norte de India y Mughlai. Al ser Bengaluru una región del sur de India llama la atención tanta presencia de cocina del norte. 

# In[69]:


top_cinco_ubicaciones= df2.location.value_counts().nlargest(n=20)
top_cinco_ubicaciones = pd.DataFrame(top_cinco_ubicaciones)
top_cinco_ubicaciones = top_cinco_ubicaciones.reset_index()


# In[70]:


top_cinco_ubicaciones.columns


# In[71]:


graph2 = sns.barplot(data= top_cinco_ubicaciones, x= 'index', y= 'location')
plt.xticks(rotation=90)
graph2.set(title = 'Ubicaciones con mayor n° de pedidos') 
plt.xlabel('Ubicaciones')
plt.ylabel('Número de pedidos') 


# De todas las ubicaciones la que más apariciones hace en cuanto a pedidos en locales ubicados en cada lugar de Bengaluru está BTM, HSR, Koramangala 5th Block y el resto. Este análisis puede ser interesante cruzarlo con tipo de comida más consumida por ubicación para tener información atractiva.

# In[72]:


grp = df2.groupby('location')
grp_df = grp.get_group('BTM')
grp_df.reset_index()
grp_df.head()


# In[73]:


btm_top= grp_df.cuisines.value_counts().nlargest(n=20)
btm_top = pd.DataFrame(btm_top)
btm_top = btm_top.reset_index()


# In[74]:


btm_top.columns


# In[75]:


grp_graph= sns.barplot(data= btm_top, x='index', y= 'cuisines')
plt.xticks(rotation=90)
grp_graph.set(title = 'Número de cocinas en la ubicación BTM') 
plt.xlabel('Tipo de cocinas')
plt.ylabel('Número de pedidos') 
plt.xticks()


# El tipo de cocina North Indian es el tipo de cocina más común en Bengaluru. Es el tipo de cocina más consumida. Seguido por Biryani y South Indian.

# In[76]:


plt.figure(figsize=(16,8))
chart= sns.barplot( x="location", y='votes', data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
chart.set(title = 'Cantidad de votos por ubicación') 
plt.xlabel('Ubicaciones')
plt.ylabel('Cantidad de votos')


# Con este gráfico podemos medir que ubicaciones han realizado mayor cantidad de votaciones y en este caso fue Church Street. Este dato puede ser útil para saber que zonas tienen de a hacer más votaciones y con ello aumentar el número de respuestas en encuestas o valoración del servicio. Lo que podría ayudar a los restaurantes a mejorar sus servicios basado en las puntuaciones que reciben.

# In[77]:


cost_top= df['cost_per_two'].value_counts().nlargest(n=20)
ct_df = cost_top.to_frame()
ct_df = ct_df.reset_index()
ct_df =ct_df.rename(columns={"index":"value"})
ct_df = ct_df.astype({'value':'int'})


# In[78]:


group_df = df
group_df = group_df.groupby('cost_per_two')
group_df = group_df.get_group(300)
group_df.reset_index()


# ## Finalizando el análisis exploratorio y en búsqueda de la detección de variables dependientes / correlativas, revisamos:

# - ¿Existe alguna relación entre el costo promedio cada dos personas y la ubicación?
# 
# *Esto podría indicarnos las zonas con los precios más elevados o con tipos de comida de alto valor.*
# 
# - ¿Existe alguna relación entre los votos de mayor nivel (4/5  o 5/5) y el tipo de comida, ubicación, tipo de restaurante?
# 
# *La respuesta podría indicarnos que exigencias debe cumplir un restaurante o local para obtener mejor nivel de puntuación.*
# 
# - ¿Será posible predecir el rango de puntuación (0/5) que tendrá un restaurante, dada su ubicación y su tipo de cocina?
# 
# *La respuesta podría indicarnos un modelo de machine learning aplicable a la industria y que permita vender datos a emprendedores de la región.*

# # Implemetanción de Machine Learning

# La idea es la de poder predecir que costo aproximado va a tener un plato dada una serie de "features". Comenzaremos analizando la correlación de los datos para ver si podremos aplicar un modelo de regresión líneal o si necesitaremos otro modelo de aprendizaje supervisado. 
# 
# Como deseamos hacer un modelo que logre aproximar un costo necesitaremos un modelo supervisado. El que creo que tendrá un buen funcionamiento (Según la imagen debajo) es Lasso o ElasticNet. 

# In[79]:


from IPython.display import Image
Image(filename='modelos.png') 


# In[80]:


#Get Correlation between different variables
corr = df.corr(method='pearson')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
df.columns


# Hay pocas variables que tengan una alta relación con 'cost_per_two', por lo que un modelo de regresión líneal no sería muy acorde a nuestra necesidad. Sin embargo, vamos a avanzar con el modelo de regresión.

# ## Hacemos el fiting

# In[81]:


df.columns
df.name.astype(str)


# > Encoding de variables

# In[82]:


#Encode the input Variables
def Encode(df):
    for column in df.columns[~df.columns.isin(['rate', 'cost', 'votes'])]:
        df[column] = df[column].factorize()[0]
    return df

df_encode = Encode(df.copy())
df_encode.head()


# > Veamos si hay correlación

# In[83]:


#Get Correlation between different variables
corr = df_encode.corr(method='kendall')
plt.figure(figsize=(12,6))
sns.heatmap(corr, annot=True)
df_encode.columns


# In[88]:


df_encode.columns


# In[89]:


from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Asignamos y a la variable a predecir
y = df_encode.cost_per_two

# Creamos X con una máscara llamada features 
features = ['online_order', 'book_table', 'votes','location','rest_type','rest_type', 'rest_type', 'rest_type', 'local_type']
X = df_encode[features]
X.head()

# Dividimos los datos para validación y entrenamiento
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=3)


# In[90]:


# Definimos el modelo
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 10000, normalize = True)
lassocv.fit(train_X, train_y)


# In[91]:


lasso = Lasso(alpha = lassocv.alpha_)
lasso.fit(train_X, train_y)
lasso_predict = lasso.predict(val_X)
lasso_mae = mean_absolute_error(lasso_predict, val_y)
lasso_mse = mean_squared_error(val_y,lasso_predict)
lasso_msle = np.log(np.sqrt(mean_squared_error(val_y,lasso_predict)))
lasso_r2score = r2_score(val_y,lasso.predict(val_X))
print("Validación con MAE para Lasso: {:,.0f}".format(lasso_mae))
print("Validación con MSE para Lasso: {:,.0f}".format(lasso_mse))
print("Validación con RMSE para Lasso: {:,.0f}".format(lasso_msle))
print("Validación con R2 Score para Lasso: {:,.3f}".format(lasso_r2score))


# Según el resultado arrojado por Lasso, no tenemos grandes posibilidades de predecir la variable y con este modelo y las features elegidas.

# In[92]:


# Define a random forest model
rf_model = RandomForestRegressor(random_state=5)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
rf_val_mse = mean_squared_error(val_y,rf_val_predictions)
rf_val_rmsle = np.log(np.sqrt(mean_squared_error(val_y,rf_val_predictions)))
rf_r2score = r2_score(val_y,rf_val_predictions)
print("Validación con MAE para Random Forest Model: {:,.0f}".format(rf_val_mae))
print("Validación con MSE para Random Forest Model: {:,.0f}".format(rf_val_mse))
print("Validación con RMSE para Random Forest Model: {:,.0f}".format(rf_val_rmsle))
print("Validación con R2 Score para Random Forest Model: {:,.3f}".format(rf_r2score))


# Un R2 Score de 0,79 es mejor que el que obtuvo Lasso, por lo tanto, esto significa que tenemos más posibilidades de predecir con  el costo que tendría comida para 2 personas en Bengaluru con este modelo que con Lasso.

# En conclusión lo que puede estar pasando con los modelos es que hay un underfitting. 
