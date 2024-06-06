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
arquivo_csv = 'peras.csv'  # caminho do arquivo
dados = pd.read_csv(arquivo_csv)  # leitura dos dados
print(dados.head())
dados.info()  # verificação se há dados nulos

# Visualização dos dados
fig = px.histogram(dados, x='qualidade', text_auto=True)
fig.write_html('histograma_qualidade.html', auto_open=True)

for feature in ['tamanho', 'peso', 'docura', 'crocancia', 'suculencia', 'maturacao', 'acidez']:
    fig = px.box(dados, x=feature, color='qualidade')
    fig.write_html(f'histograma_{feature}.html', auto_open=True)

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
k_values = range(3, 16, 2)  # K ímpares entre 3 e 15
results = []
best_val_accuracy = 0
best_k = 0

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia na validação com k = {k}: {val_accuracy}")
    results.append((k, val_accuracy))
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_k = k

# Treinamento do modelo final
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(normalizacao.transform(x_trainval), y_trainval)

# Avaliação do modelo no conjunto de teste
y_test_pred = final_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Acurácia no teste com melhor k = {best_k}: {test_accuracy}")

# Salvando o modelo
joblib.dump(final_model, 'melhor_modelo_knn.pkl')

# Imprimir todos os resultados
for k, val_acc in results:
    print(f"k: {k}, Acurácia de Validação: {val_acc:.4f}")
