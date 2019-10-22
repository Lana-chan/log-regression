#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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

# vetor de probabilidades 
probability_test = logreg.predict_proba(X_test)
probability_train = logreg.predict_proba(X_train)
print(probability_test)

# testando o modelo 
print(classification_report(y_train, Ptrain))
print(classification_report(y_test, Ptest))

%matplotlib inline

# calculando o score da predição
score = logreg.score(X_test,y_test)
print('Score: ', score)

# Testando valores específicos no modelo com a confusion matrix
cf = confusion_matrix(y_test, Ptest)
accuracy = accuracy_score(y_test, Ptest)
print("Confusion Matrix: ", cf)
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,Ptest)))
plt.show()
print("Accuracy: ", accuracy)


# Cross-validation
cross_val = cross_val_score(LogisticRegression(solver='liblinear'), X,y, scoring='accuracy', cv=10)
print('Cross-Validation: ', cross_val)
print('Cross-Validation mean: ', cross_val.mean())

# normalizando X_test
df_norm = pd.DataFrame(StandardScaler().fit_transform(X_test))

# probabilidade de ser 1
probability_test_df = pd.DataFrame(probability_test)
probability_t = pd.DataFrame(probability_test_df[0])

# plotando o gráfico de probabilidades e colunas

plt.scatter(df_norm[0],probability_test_df[1])
plt.scatter(df_norm[1],probability_test_df[1])
plt.scatter(df_norm[2], probability_test_df[1])
plt.scatter(df_norm[3], probability_test_df[1])

# variaveis que podem ser desprezadas pro modelo

plt.scatter(df_norm[4],probability_test_df[1])
plt.scatter(df_norm[5],probability_test_df[1])
plt.scatter(df_norm[6],probability_test_df[1])


def log_odds(p):
    return np.log(p / (1 - p))

log_odds_prob = log_odds(probability_test_df[1])

plt.plot(df_norm[0], log_odds)


