import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib


# Função para processar o arquivo CSV e calcular previsões usando KNN
def executa(file_path, k):
    # Processar o arquivo
    dados = pd.read_csv(file_path)

    if 'qualidade' in dados.columns:
        raise ValueError("O arquivo CSV não deve conter a coluna 'qualidade'.")

    x = dados[['id','tamanho', 'peso', 'docura', 'crocancia', 'suculencia', 'maturacao', 'acidez']]

    geraModeloKNNUsuario(k)

    modelo_knn = joblib.load('modelo_knn_usuario.pkl')

    # Fazer previsões com o modelo KNN
    y_pred_knn = modelo_knn.predict(x)

    # Converter previsões para rótulos 'boa' ou 'ruim'
    label_encoder = LabelEncoder()
    label_encoder.fit(['ruim', 'boa'])
    y_pred_knn_label = label_encoder.inverse_transform(y_pred_knn)

    # Adicionar previsões ao DataFrame
    dados['qualidade'] = y_pred_knn_label

    # Salvar o DataFrame com as previsões como CSV
    result_filename = 'result_' + os.path.basename(file_path)
    result_file_path = os.path.join(os.path.dirname(file_path), result_filename)
    dados.to_csv(result_file_path, index=False)

    return result_file_path

def geraModeloKNNUsuario(kUsuario):

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
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval,
                                                      test_size=val_size / (train_size + val_size), random_state=5)

    # Normalizando os dados
    normalizacao = MinMaxScaler()
    x_train = normalizacao.fit_transform(x_train)
    x_val = normalizacao.transform(x_val)
    x_test = normalizacao.transform(x_test)

    # Treinamento do modelo final
    final_model = KNeighborsClassifier(n_neighbors=kUsuario)
    final_model.fit(normalizacao.transform(x_trainval), y_trainval)

    # Avaliação do modelo no conjunto de teste
    y_test_pred = final_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Acurácia no teste com melhor k = {kUsuario}: {test_accuracy}")

    # Salvando o modelo
    joblib.dump(final_model, 'modelo_knn_usuario.pkl')

# Exemplo de uso
if __name__ == '__main__':
    # Parâmetros de exemplo
    file_path = 'peras2.csv' #Breno aqui tu vai receber um arquivo .csv
    k_value = 3 #Breno aqui tu vai receber o valor de K q o usuario informar

    # Processar o CSV e obter o caminho do arquivo resultante
    try:
        result_file_path = executa(file_path,k_value)
        print(f"Arquivo resultante salvo em: {result_file_path}")
    except ValueError as e:
        print(f"Erro: {e}")