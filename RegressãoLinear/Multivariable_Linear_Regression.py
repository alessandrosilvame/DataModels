# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:09:20 2019

@author: Alessandro
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

sns.set(style='darkgrid')

Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }

df = pd.DataFrame(Stock_Market)

df

df.describe()

df.head(10)

df['Interest_Rate'].corr(df['Unemployment_Rate'])

df.corr(method='spearman')  # Spearman Method
df.corr()                   # Pearson Method

plt.scatter(df['Interest_Rate'],df['Unemployment_Rate'], color = 'red')
plt.title('Interest_Rate Vs Unemployment_Rate', fontsize = 14)
plt.grid(True)
plt.show()

plt.boxplot(df['Interest_Rate'])
plt.title('Diagrama de Caixa - Interest_Rate')
plt.grid(True)
plt.show()

sns.boxplot(df['Interest_Rate'], orient='v', color='LightBlue')

X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['Stock_Index_Price']

regr = linear_model.LinearRegression()

regr.fit(X,Y)

regr.intercept_ #Intercepto
regr.coef_      # Coeficiente


#Prevendo resultados
New_Interest_Rate = float(input('Digite o valor no novo Interest_Rate:'))
New_Unemployment_Rate = float(input('Digite o valor no novo Unemployment_Rate:'))

New = regr.predict([[New_Interest_Rate,New_Unemployment_Rate]])

print('O Stock_Index_Price predito é:\n', New)







