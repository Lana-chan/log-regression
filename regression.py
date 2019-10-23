#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv('auto-mpg1.csv')

# alteramos o atributo mpg para se este é maior ou igual que 23 ou não
data['mpg'] = data['mpg'].apply(lambda x: 1 if x>=23.0 else 0)

# removemos entradas desconhecidas
data.replace('?', np.nan, inplace=True)
data = data.dropna()

# X são as variáveis independentes, y é a variável dependente
cols = ['cilindros', 'cilindradas', 'cavalos', 'peso', 'aceleração', 'ano', 'origem']
X = data[cols]
y = data['mpg']

# dividimos algumas entradas para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# aplicamos modelo de regressão logística
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)

# resultados do modelo
Ptrain = logreg.predict(X_train)
Ptest = logreg.predict(X_test)
print(classification_report(y_train, Ptrain))
print(classification_report(y_test, Ptest))

%matplotlib inline

# Leave-one-out Cross-validation
# também conhecido como teste de jackknife
# (reamostragem por retirar uma observação)
from sklearn.model_selection import LeaveOneOut, cross_val_score
loo = LeaveOneOut()
scores = cross_val_score(logreg, X, y, cv=loo)
# precisão do modelo e o intervalo de confiança
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 1.96))
# notamos que leave-one-out/jackknife é um estimador que resulta em uma grande variância
# https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out-loo
# "In terms of accuracy, LOO often results in high variance as an estimator for the test error."

#%%
