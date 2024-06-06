import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Definindo as proporções para treinamento, validação e teste
train_size = 0.6
val_size = 0.2
test_size = 0.2

# Carregando os dados
arquivo_csv = 'MelhorModelo/peras.csv'  # caminho do arquivo
dados = pd.read_csv(arquivo_csv)  # leitura dos dados
print(dados.head())
dados.info()  # verificação se há dados nulos

# Preparação dos dados
x = dados.drop('qualidade', axis=1)
y = dados['qualidade']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Divisão dos dados
x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=test_size, random_state=5)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=val_size/(train_size + val_size), random_state=5)

# Normalizando os dados
normalizacao = MinMaxScaler()
x_train = normalizacao.fit_transform(x_train)
x_val = normalizacao.transform(x_val)
x_test = normalizacao.transform(x_test)

# Treinamento e validação do modelo
kUsuario = 3 # k INFORMADO NA TELA PELO USUÁRIO

# Treinamento do modelo final
final_model = KNeighborsClassifier(n_neighbors=kUsuario)
final_model.fit(normalizacao.transform(x_trainval), y_trainval)

# Avaliação do modelo no conjunto de teste
y_test_pred = final_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Acurácia no teste com melhor k = {kUsuario}: {test_accuracy}")

# Salvando o modelo
joblib.dump(final_model, 'modelo_knn_usuario.pkl')