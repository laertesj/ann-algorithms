# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:20:53 2021

@author: eitop
"""

import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
 
X, y = datasets.load_iris(return_X_y=True)
 
# dividindo o conjunto de dados em treinamento, teste e validação
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
 
# aplicando feature scaling (normalizaço 1, -1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)
 
# Transformamos os valores da variável de saída para categórico para utilizarmos na compilação da rede
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_val = le.fit_transform(y_val)

y_train = keras.utils.to_categorical(y_train, num_classes = 3)
y_test = keras.utils.to_categorical(y_test, num_classes = 3)
y_val = keras.utils.to_categorical(y_val, num_classes =3 )

# Construindo a rede
model = Sequential()
# Chamada da camada de entrada (input_dim) e camada oculta
model.add(Dense(12, input_dim=4))
# Camada de saída
model.add(Dense(3, activation='sigmoid'))
 
# Compilando a rede
# O parâmetro 'loss' realiza o cálculo de erro a partir da soma do erro quadrático, ou seja, o MSE, aqui só possui outro nome
# como a saída do nosso problema não é binária, possuimos mais de 2 classes, precisamos utilizar esse parâmetro 'categorical_crossentropy'
# por isso a necessidade de transformar a variável de saída para categorica7
_sgd = SGD(lr=0.05)
model.compile(optimizer=_sgd, loss='mean_squared_error', metrics=['accuracy'])
    
# Treinando a rede com o conjunto de treinamento e validando com o conjunto de validação
history = model.fit(X_train, y_train, batch_size=32, epochs=500, validation_data=(X_val, y_val))
 
# Mostrando a acurácia do modelo a partir do conjunto de validação
"""test_acc = model.evaluate(X_val, y_val)
print('Test accuracy:', test_acc)
print()"""

print()
 
# Plotando o gráfico de erro
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,501)
plt.plot(epochs, loss_train, 'g', label='Training Loss')
plt.plot(epochs, loss_val, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Iterações')
plt.ylabel('MSE')
plt.legend()
plt.show()
  
# Prevendo os resultados para o conjunto de teste
print("\n\nValidando o Modelo com o conjunto de teste (Matriz de Confusão):\n")
y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
 
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))