# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:57:38 2019

@author: amanoels
"""
# Importando Bibliotecas
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('seaborn') # Setando padrão visual do Seaborn
import pandas as pd
import statsmodels.api as sm

# Gerando Data Frame
df = pd.read_excel('Sample - Superstore.xls')

# Filtrando dados somente referentes a categoria móveis
furniture = df.loc[df['Category'] == 'Furniture']

# Verificando massa de dados
furniture['Order Date'].min(), furniture['Order Date'].max()

# Pré Processamento dos Dados
# Etapa onde removo as colunas que não são necessarias, 
# Valores faltantes, agregar vendas por datas e outras.
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID',
        'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code',
        'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
        'Quantity', 'Discount', 'Profit']

# Removendo colunas selecionadas de forma implicita
furniture.drop(cols, axis=1, inplace=True)

# Ordenando valores por data do pedido
furniture = furniture.sort_values('Order Date')

# Contagem de nulos por Coluna
furniture.isnull().sum()

# Agrupando os dados por Data do Pedido
furniture = furniture.groupby(['Order Date']).sum()

# Checando os indices das colunas
furniture.index

# Reordenando os dados para Mensal obtendo a média por Mês
y = furniture['Sales'].resample('MS').mean()

# Exemplo do resultado, apresentando dados de 2017 mensalmente
y['2017':]

# Apresentando os dados de 2017 de forma grafica via Matplotlib
y.plot(figsize=(15,6))
plt.show()

# Quando os dados são plotados, fica mais facil de ver os padrões que aparecem
# Neste caso, no começo do ano as vendas caem e no fim de ano elas sobem 
# Isto se repete todos os anos apresentados

# Gerando decomposição (Mostrando padrõa, ruido e sazonalidade)
decomposition = sm.tsa.seasonal_decompose(y, model='addytive')
fig = decomposition.plot()
plt.show()

# Desenvolvimento do modelo ARIMA
p = d = q = range(0,2)

# Preparação dos iteradores
pdq = list(itertools.product(p,d,q))

# Geração de Sazonalidade
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in 
                list(itertools.product(p,d,q))]

print('Exemplos de combinações para sazonalidade ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[4]))

# Busca no grid gerado pelo melhor modelo ARIMA
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# Melhor modelo baseado no AIC : 297.78
# Passando os parametros para o ARIMA
mod = sm.tsa.statespace.SARIMAX(y, 
                                order=(1,1,1),
                                seasonal_order=(1,1,0,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
# Ajustando o modelo
results = mod.fit()
# Apresentando os resultaods do ajuste do modelo
print(results.summary().tables[1])

# Rodando o diagnóstico para avaliar possiveis desvios
results.plot_diagnostics(figsize=(16,8))
plot.show()

# O mesmo não estsa perfeito mas os residuos estão contidos

# Gerando as previsões
# Para ajudar a validar a precisão do modelo, iremos verificar os dados gerados
# pela previsão, contra os dados de 2017

# Gerando as previsões iniciando em 01-01-2017
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)


# Plotando dados de 2014
ax = y['2014':].plot(label='Observado')
# Gerando a previsão com dados previstos e existentes
pred.predicted_mean.plot(ax=ax, label='Previsão um passo a frente', alpha=0.7, figsize=(14,7))
# Gerando intervalos de inteiros
pred_ci = pred.conf_int()
# Preparando o plot
ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1], color='k', alpha=.2)
ax.set_xlabel('Data')
ax.set_ylabel('Venda de Móveis')
plt.legend()
plt.show()

# Modelo de previsão capturando bem as tendencias de queda no inicio e aumento no fim 
# ano bem como a sazonaidade desta. 

# Gerando previsão futura
y_forecast = pred.predicted_mean

# Iniciando em 2017-01-01
y_truth = y['2017-01-01':]

# Calculando o Main Squared Error ( Erro ao Quadrado)
mse = ((y_forecast - y_truth)**2).mean()

# Apresentando o valor do erro ao quadrado
print('O valor obtido para o MSE ( Erro ao Quadrado) é:{}'.format(round(np.sqrt(mse),2)))
# Quanto menor o erro mais perto de obter uma linha perfeita 
# Como o range de vendas varia de 400 a 1200, uma diferença prevista de 151.64 esta muito boa

# Gerando as previsões e apresetando os resultados, 100 passos a frente
pred_uc = results.get_forecast(steps=100)
# Gerando os intervalos inteiros 
pred_ci = pred_uc.conf_int()

# Gerando o grafico
ax = y.plot(label='Observado', figsize=(14,7))
pred_uc.predicted_mean.plot(ax=ax, label='Previsto')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1], color='k', alpha=.25) 
ax.set_xlabel=('Data')
ax.set_ylabel=('Venda Móveis')
plt.legend()
plt.show()

# Claramente o modelo capturou a sazonalidade e a tendencia da serie prevista 
# Claro que com o passar do tempo a tendencia é a margem de erro aumentar 
# para para os primeiros anos, esta bem proximo do real








                                








    











