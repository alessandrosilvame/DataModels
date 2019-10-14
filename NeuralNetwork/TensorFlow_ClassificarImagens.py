# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:41:53 2019

@author: amanoels
"""

from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librarie sauxiliares
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Baixando pacote fashion_mnist
fashion_mnist = keras.datasets.fashion_mnist

# Carregando dados para treinamento e testes
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""
Label 	Classe
0 	    Camisetas/Top (T-shirt/top)
1 	    Calça (Trouser)
2 	    Suéter (Pullover)
3 	    Vestidos (Dress)
4 	    Casaco (Coat)
5 	    Sandálias (Sandal)
6 	    Camisas (Shirt)
7 	    Tênis (Sneaker)
8 	    Bolsa (Bag)
9 	    Botas (Ankle boot) """

# A base de dados não traz os labels, sendo assim estão sendo armazenados aqui 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shape para determinar o dimensionamento
train_images.shape

# Labels com um conjunto de treinamento 
len(train_labels)

# Cada label é um inteiro entre 0 e 9 
train_labels

# Existem 10000 imagens no cojunto de teste
test_images.shape

# labels no conjunto de teste
len(test_labels)

# Os dados precisam ser pré-processados antes de treinar a rede. Verificando a primeira
# imagem percebemos que os pixels estão entre 0 e 255. 
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Estes valores precisam ser escalados no intervalor entre 0 e 1 antes de alimentar 
# o modelo da rede neural para tal os valores devem ser divididos por 255. 
train_images = train_images / 255.0
test_images = test_images / 255.0

# Para garantir que os dados estão no formato correto e prontos para construir e treinar
# a rede, apresento as 25 primeiras imagens. 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# As figuras estão sendo apresentadas e montadas corretamente, desta forma começo a construção 
# do modelo 

# Duas etapas principas
# 1 Montar as camadas
# 2 Compilar as camadas

# Modelagem
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
])
    
# Compilação ( Função loss mede o erro, o otimizador aponta para o caminho correto, 
# baseado nos resultados de loss e a métrica de medição é a acuracia. 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
model.fit(train_images, train_labels, epochs=50)

# Avaliação do modelo e sua acuracia. 
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# A acuracio não esta boa ainda, precisa ser ajustado, pois o modelo esta com overfiting

# Prevendo classificação
predictions = model.predict(test_images)

# Primeira posiçao da previsão, é um array de 10 posições
predictions[0]

# Buscando a maior confiana exitente neste array
np.argmax(predictions[0])
# Desta forma o modelo esta confiante de que esta imagem é uma "Ankle boot".

# Para validar que a classe é mesmo esta, vou checar a massa de testes
train_labels[0]

# Função de preparação das imagens
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
 
# Apresentação de informação
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Plotando a primeira previsão 
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Previsão para uma unica imagem
img = test_images[0]
print(img.shape)


    




        
    
            





   
