#Setup'de bibliotecas
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

#Importação do dataset
wine = datasets.load_wine()

# Criando o DataFrame
df_wine = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df_wine['class'] = wine.target

print('\n',df_wine['class'].value_counts(dropna = False))

# Dividindo os dados em treino e teste:
X_train, X_test, y_train, y_test = train_test_split(df_wine.drop('class',axis=1),df_wine['class'],test_size=0.2, random_state=42)

print(X_train.shape,X_test.shape)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

resultado_teste = clf.predict(X_test)
resultado_treino = clf.predict(X_train)

#Validação do modelo
print('Teste',metrics.classification_report(y_test,resultado_teste))
print('Treino',metrics.classification_report(y_train,resultado_treino))

#Árvore de decisão em forma de texto
text_representation = tree.export_text(clf)
print(text_representation)

#Árvore de decisão em forma de diagrama
tree.plot_tree(clf,filled=True)
plt.show()



