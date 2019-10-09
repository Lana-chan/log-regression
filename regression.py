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
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# resultados do modelo
Ptrain = logreg.predict(X_train)
Ptest = logreg.predict(X_test)
print(classification_report(y_train, Ptrain))
print(classification_report(y_test, Ptest))

%matplotlib inline

#%%
