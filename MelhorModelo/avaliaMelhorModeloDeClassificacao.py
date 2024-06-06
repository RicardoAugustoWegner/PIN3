import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Definindo as proporções para treinamento, validação e teste
train_size = 0.6
val_size = 0.2
test_size = 0.2

# Carregando os dados
arquivo_csv = 'peras.csv'  # caminho do arquivo
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

# Carregar os modelos
modelo_arvore = joblib.load('modelo_arvore_decisao.pkl')
modelo_knn = joblib.load('modelo_knn.pkl')

# Fazer previsões e avaliar os modelos
y_test_pred_arvore = modelo_arvore.predict(pd.DataFrame(x_test, columns=x.columns))

# Normalizando os dados apenas para o modelo KNN
normalizacao = MinMaxScaler()
x_test_norm = normalizacao.fit_transform(x_test)

y_test_pred_knn = modelo_knn.predict(x_test_norm)

accuracy_arvore = accuracy_score(y_test, y_test_pred_arvore)
accuracy_knn = accuracy_score(y_test, y_test_pred_knn)

print(f"Acurácia no teste - Modelo de Árvore de Decisão: {accuracy_arvore}")
print(f"Acurácia no teste - Modelo KNN: {accuracy_knn}")

melhor_modelo = "Árvore de Decisão" if accuracy_arvore > accuracy_knn else "KNN"
melhor_acuracia = max(accuracy_arvore, accuracy_knn)
segundo_modelo = "KNN" if melhor_modelo == "Árvore de Decisão" else "Árvore de Decisão"
segunda_acuracia = min(accuracy_arvore, accuracy_knn)
diferenca = round(abs(accuracy_arvore - accuracy_knn), 5)

print(f"Melhor modelo: {melhor_modelo} - Acurácia: {melhor_acuracia}")
print(f"Diferença para o segundo modelo ({segundo_modelo}): {diferenca}")