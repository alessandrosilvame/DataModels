# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:47:58 2019

@author: amanoels
"""

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('precision', 2)


features = pd.read_csv('temps.csv')
features.head(5)
features.describe()

features1 = features.iloc[:,:8]

features1.head(5)

features1['friend'] = features.iloc[:,11:12]

features1.head(5)

""" 
Descritivo das variaveis

Year = Ano de 2016 - Dados Coletados para todos os pontos
month = numero do mês para o ano
day = numero de dia para o ano
week = dia de semana em formato string
temp_2 = Temperatura maxima 2 dias antes
temp_1 = Temperatura maxima 2 dia antes
average = média historica da temperatura maxima
actual = medição da temperatura maxima
friend = a previsão do seu amigo, um número aleatório entre 20 abaixo da média e 20 acima da média

"""

print('A quantidade de linhas e colunas da amostra é de : \n', features1.shape)

features1.describe()

plt.plot(features1['month'], features1['average'])

# Transformando variaveis categóricas em numeros

features2 = pd.get_dummies(features1)
features2.iloc[:,5:].head(5)
features2.describe()

features2.shape

import numpy as np

# Valores que queremos prever
labels = np.array(features2['actual'])

# Removendo os Labels dos recursos
# axis 1 se refere as colunas
features = features2.drop('actual', axis=1)

# Salvando os nomes para uso posterior
feature_list = list(features2.columns)

# Convertendo em um array numpy
features = np.array(features)

# importando pacotes para dividir a massa entre trainamento e teste
from sklearn.model_selection import train_test_split

# Dividindo os dados entre treinamento e testes
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Analise de divisão da amostra
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# É sempre esperado que o numero de colunas e linhas de train_Features batam com o numero
# de test_features e train_labels com test_labels

""" Antes de iniciar o processo de medição, é necessario estabelecer uma linha base, 
algo que esperamos melhorar com o nosso modelo, e caso não consigamos, ou utilizamos
outro modelo ou admitimos que o Machine Learning não é a solução para este problema. 
Neste caso podemos utilizar a maximo da média histórica. Em outras palavras, 
a linha base será o erro que obteriamos se simplismente medissemos a média maxima 
de temperatura para todos os dias. """

# A linha base de previsão será a média histórica
baseline_preds = test_features[:, feature_list.index('average')]

# Linha base de erros, e apresentação da linha base de erros
baseline_errors = abs(baseline_preds - test_labels)
print('Linha Base Média de Erros é:\n', round(np.mean(baseline_errors), 2))

# Importando a biblioteca do modelo que será utilizado
from sklearn.ensemble import RandomForestRegressor

# Instanciando o modelo com 1000 arvores 
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Treinando o modelo com dados de treinamento
rf.fit(train_features, train_labels)

# Ao fazer a previsão a seguir estou interessado em quão longe a previsção do 
# modelo esta do erro absoluto visto anteriormente que tinha a média de 5.06

# Utlizando o modelo treinado para fazer previsões baseados nos dados de teste
predictions = rf.predict(test_features)

# Calculando o erro absoluto
erros = abs(predictions - test_labels)

# Exibição do erro absoluto
print('A média do erro absoluto previsto é:\n', round(np.mean(erros), 2), 'Graus')
# Resultdo de 3.83 Graus uma melhora de quase 25% em relação a linha base original

""" Determinando a performance da medição """

# Para colocar em perspectiva vamos determinar a accuracia  (MAPE)
# MAPE = MEAN ABSOLUTE PERCENTAGE ERROR
mape = 100 * (erros / test_labels)

# Calculando e exibindo a precisão do modelo
accuracy = 100 - np.mean(mape)
print('Precisão ou Acuracia:\n', round(accuracy, 2),'%.')

""" O modelo aprendeu como prever a temperatura para o dia de amanhã em Seattle 
com previsão de 94% de acertividade, é um resultado muito bom"""

""" É possivel utilizar ferramentas dentro do proprio Scikit-Learn
para melhorar os hyperparametros do modelo, mas este é um processo bem longo. 
É sempre importante manter em perspectiva que o seu primeiro modelo nunca, nunca, 
deve ser utilizado em produção devem ser gerados pelo menos 3 modelos, de preferencia
com amostras diferentes da mesma base de dados para determinar a qualidade de um 
modelo e seus hyperparametros"""

""" Determinado que o modelo é bom, precisamos descobrir como reportar para uma 
visão humnada, e sendo assim, temos 2 caminhos, podemos checar 1 arvore na floresta
ou podemos verificar as importancias das variaveis explicativas. """

# Visualizando 1 Arvore
from sklearn import tree
import pydotplus
from IPython.display import Image

# Colocando 1 arvore da floresta para visualização
arvore = rf.estimators_[5]

# Gerando arquivo dot
dot_data = tree.export_graphviz(arvore, out_file = None,
                                filled = True, 
                                rounded = True, 
                                special_characters = True)

# Preparando ao leitura do arquivo dot para imagem
graph = pydotplus.graph_from_dot_data(dot_data)

# Exibindo a imagem
Image(graph.create_png())

""" Ficou uma imagem bem grande, com 15 camadas, porem é meio dificil analisar
sendo assim, vamos limitar a 3 camadas de profundidade para uma analise mais 
detalhada"""

# Limitando a profundidade
# Gerando o modelo
rf_small = RandomForestRegressor(n_estimators = 10, max_depth = 3)

# Treinando o modelo
rf_small.fit(train_features, train_labels)

# Extraindo uma arvore pequena 
tree_small = rf_small.estimators_[5]

# gerando a visualização a pequena arvore
# Gerando arquivo dot
dot_data = tree.export_graphviz(tree_small, out_file = None,
                                filled = True, 
                                rounded = True, 
                                special_characters = True)

# Preparando ao leitura do arquivo dot para imagem
graph = pydotplus.graph_from_dot_data(dot_data)

# Exibindo a imagem
Image(graph.create_png())

""" Com esta arvore se torna mais simples de conseguir determinar o caminho que os 
fluxo esta seguindo e nas folhas mais simples de encontrar a previsão"""

# Determinando as variavies imporantes no seu modelo
importances = list(rf.feature_importances_)

# Lista de variaveis importantes
feature_importances = [(feature, round(importance, 2)) for feature, 
                       importance in zip(feature_list, importances)]

# Ordenando pela mais importante para menos importante
feature_importances = sorted(feature_importances, key = lambda x: x[1], 
                             reverse = True)

[print('Variaveis: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

""" Com isto vemos que as duas variaveis mais importantes são Temp_1 e Average"""

# Criando um modelo sem as variaveis que não apoiam a previsão
rf_most_important = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Extraindo as variaveis mais importantes
important_indices = [feature_list.index('temp_1'), feature_list.index('average')]

train_important = train_features[:,important_indices]
test_important = test_features[:, important_indices]

# Treinando o modelo de RandomForest
rf_most_important.fit(train_important, train_labels)

# Fazendo as previsões e determinando o erro
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)

# Apresentando a performance da métrica
print('Erro médio absoluto:\n', round(np.mean(errors),2), 'Graus')

# gerando o MAPE 
mape = np.mean(100*(errors / test_labels))
accuracy = 100 - mape

print('Precisão é:\n', round(accuracy, 2), '%.')

""" Com isto podemos afirmar que precisamos apenas destas 2 variaveis para
realizar de forma satisfatória esta previsão"""

# Gerando um grafico de barras para facilidade de visualizaçao
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
# Grafico para apresentar a importancia das varivaveis
x_values = list(range(len(importances)))# Fazendo o grafico de barras
plt.bar(x_values, importances, orientation = 'vertical')# Tick labels for x 
plt.xticks(x_values, feature_list, rotation='vertical')# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

""" Gerando um grafico com as previsão x valor atual"""
#Use datetime for creating date objects for plotting
import datetime# Dates of training values
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');


# Visão Geral

# Make the data accessible for plotting
true_data['temp_1'] = features[:, feature_list.index('temp_1')]
true_data['average'] = features[:, feature_list.index('average')]
true_data['friend'] = features[:, feature_list.index('friend')]# Plot all the data as lines
plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)# Formatting plot
plt.legend(); plt.xticks(rotation = '60');# Lables and title
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');


















