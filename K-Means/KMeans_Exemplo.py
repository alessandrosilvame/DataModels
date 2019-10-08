# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:46:35 2019

@author: amanoels
"""

""" Exemplo de algoritimos K-Means """

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

# Gerando os dados para analise
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apresentando os dados gerados
plt.scatter(X[:,0], X[:,1])

# Preparando o wcss para receber os valores de inertia_
wcss = []

# Loop para analise do Elbow Method, permitindo encontrar a quantidade de cluster mais 
# adequada para a analise
for i in range (1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Apresentando os resultados de forma grafica
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Numero de Clusters')
plt.ylabel('WCSS')
plt.show()

# Após encontrar o valor correto para os clusters (4) sigo para a previsão
# Parametro setado em n_clusters
# O parametro 'k-means++', faz com que a inicialização não seja automatica, evitando problemas de performance
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Ajustando o modelo
y_pred = kmeans.fit_predict(X)

# Gerando grafico com a massa de dados e sobrepondo os cluster gerados em vermelho
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='green')
plt.show()





