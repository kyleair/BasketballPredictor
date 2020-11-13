import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#data preperation
df = pd.read_csv('allwinloss.csv')


x = df.iloc[:,0:16]
y = df.iloc[:,16]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model building
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
mlp.fit(x_train, y_train.values.ravel())

#results analysis

predictions = mlp.predict(x_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))