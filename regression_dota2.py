#%%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# nomes dos personagens de dota2
with open("data/heroes.json",'r') as f: 
	heroes = json.load(f) 
	heroes = [d['name'] for d in heroes['heroes']]

# monta nomes de colunas
colunas = ['won', 'cluster', 'game_mode', 'game_type'] + heroes

# dataset
train_data = pd.read_csv('data/dota2Train.csv', names=colunas, index_col=False)
test_data = pd.read_csv('data/dota2Test.csv', names=colunas, index_col=False)

# alteramos o resultado de perda de -1 para 0
train_data['won'] = train_data['won'].apply(lambda x: 1 if x==1 else 0)
test_data['won'] = test_data['won'].apply(lambda x: 1 if x==1 else 0)

# X são as variáveis independentes, y é a variável dependente
cols = ['game_mode', 'game_type'] + heroes
X_train = train_data[cols]
X_test = test_data[cols]
y_train = train_data['won']
y_test = test_data['won']

# aplicamos modelo de regressão logística
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# resultados do modelo
Ptrain = logreg.predict(X_train)
Ptest = logreg.predict(X_test)
print(classification_report(y_train, Ptrain))
print(classification_report(y_test, Ptest))

%matplotlib inline

#%%
