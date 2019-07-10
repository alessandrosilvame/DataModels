# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:23:50 2019

@author: amanoels
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt # Import Biblioteca de Visualização
import seaborn as sns
from sklearn.linear_model import LinearRegression
#from plotnine import * 


#Importando dados com Pandas
df = pd.read_excel("D:\DataSets_Exemplos\RLSBEZERRO.xlsx")

# Convertendo para arrays do Numpy
#df = np.array(df)
# Fazendo o Reshape em x pois é necessario um array bi-dimensional
x = np.array(df['Medida(CM)']).reshape(-1,1)
y = np.array(df['Peso (KG)'])

sns.set()
# Gerando um grafico de dispersão
#plt.scatter(x,y)
sns.scatterplot(x,y)
#plt.scatter(df['Medida(CM)'],df['Peso (KG)'])

# Gerando analise de Boxplot
#plt.boxplot(x)
#plt.label("Diagrama de Caixa Medida CM")
sns.boxplot(x, orient = "v", notch = "TRUE")
sns.boxplot(y, orient = "v", notch = "TRUE")

# Gerando o modelo
model = LinearRegression()

# Carregando o modelo
model.fit(x,y)

# Obtendo os resultados
r_sq = model.score(x, y)
print('O r² de Determinação é :', r_sq) # Precisão do modelo
print('O Intercepto do modelo é :', model.intercept_) # Intercepto do modelo
print('O Coeficiente do modelo é : ', model.coef_) # Medida CM do modelo
# Prevendo um resultado
# Gerando uma analise de teste
a = {'Peso_Previsto':[20,28]}
a
# Transformando em DataFrame
b = pd.DataFrame(a)
b
# Agrupando resultados em uma estrutura para predição
ajuste = model.intercept_ + model.coef_ * b
# Exibindo
ajuste

# Gerando resultado em tempo real
dados = int(input('Digite a medida do torax em CM do Bezerro: ' ))
# Calculando resultado do modelo.
ajuste2 = model.intercept_ + model.coef_ * dados
# Exibindo resultado. 
print('A medida CM informada foi:',dados,'CM',
      'e a previsão de peso do Bezerro é de :', ajuste2,'KG')









































#geom_boxplot(aes(x, stat='boxplot', show_legend = "True")
# Convertendo para DataFrame com nome de 
#df_x = pd.DataFrame(x, columns = ['Medida CM'])
#ggplot(df, aes(x = 'Medida(CM)', y = 'Peso (KG)')) + geom_boxplot()
#ggplot(df, aes(x = 'Medida(CM)', y = 'Peso (KG)')) + geom_point()

#teste = df.iloc[:, 1:3]

#plt.scatter(teste['Preco_Anunciado'],teste['Area_Util'])






