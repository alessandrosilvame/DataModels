# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:01:13 2019

@author: amanoels
"""
# Importando as bibliotecas
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
# Padrão de estetica
sns.set()

# Carregando o DataSet para analise
breast_cancer = load_breast_cancer()

# Criando DataFrame com os dados do DataSet
X = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)

# Mantendo apenas as colunas desejadas para analises
X = X[['mean area', 'mean compactness']]

# Categorizando colunas alvo
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
y = pd.get_dummies(y, drop_first = True)

# Definindo massa de testes
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

# Gerando o modelo
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Ajustando o modelo
knn.fit(X_train, y_train)

# Prevendo se um tumor é benigno ou maligno baseado na Compactness e Area
y_pred = knn.predict(X_test)

# Apresentação dos dados de Teste
sns.scatterplot(
        x='mean area', 
        y='mean compactness',
        hue='benign',
        data=X_test.join(y_test, how='outer'))

# Apresentação dos dados previstos
plt.scatter(
        x=X_test['mean area'],
        y=X_test['mean compactness'],
        c=y_pred,
        cmap='coolwarm',
        alpha=0.7)

# Avaliação da matriz de confusão
c = confusion_matrix(y_test, y_pred)

# Precisão do modelo 
d = pd.DataFrame(c)
accuracy = (d.iloc[0,0] + d.iloc[1,1]) / np.ma.count(y_pred)
accuracy
# 0,84% de precisão do modelo









