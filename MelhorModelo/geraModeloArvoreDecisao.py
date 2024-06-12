import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=val_size/(val_size + train_size), random_state=5)

# Treinamento e validação do modelo
depth_range = range(3, 9)
results = []
best_val_accuracy = 0
best_depth = 0

for depth in depth_range:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia na validação com profundidade {depth}: {val_accuracy}")
    results.append((depth, val_accuracy))
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_depth = depth

# Treinamento do modelo final
final_model = DecisionTreeClassifier(max_depth=best_depth)
final_model.fit(x_trainval, y_trainval)

# Avaliação do modelo no conjunto de teste
y_test_pred = final_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Acurácia no teste com melhor profundidade {best_depth}: {test_accuracy}")

# Salvando o modelo
joblib.dump(final_model, 'modelo_arvore_decisao.pkl')
