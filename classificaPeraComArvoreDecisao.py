import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Função para processar o arquivo CSV e calcular previsões com árvore de decisão
def executa(file_path):
    # Processar o arquivo
    dados = pd.read_csv(file_path)

    if 'qualidade' in dados.columns:
        raise ValueError("O arquivo CSV não deve conter a coluna 'qualidade'.")

    x = dados

    # Carregar o modelo de árvore de decisão pré-treinado
    modelo_arvore = joblib.load('MelhorModelo/modelo_arvore_decisao.pkl')

    # Fazer previsões com o modelo de árvore de decisão
    y_pred_arvore = modelo_arvore.predict(x)

    # Converter previsões para rótulos 'boa' ou 'ruim'
    label_encoder = LabelEncoder()
    label_encoder.fit(['ruim', 'boa'])
    y_pred_arvore_label = label_encoder.inverse_transform(y_pred_arvore)

    # Adicionar previsões ao DataFrame
    dados['qualidade'] = y_pred_arvore_label

    # Salvar o DataFrame com as previsões como CSV
    result_filename = 'result_' + os.path.basename(file_path)
    result_file_path = os.path.join(os.path.dirname(file_path), result_filename)
    dados.to_csv(result_file_path, index=False)

    return result_file_path


# Exemplo de uso
if __name__ == '__main__':

    # Parâmetros de exemplo
    file_path = 'peras1.csv' #Breno aqui tu vai enviar como parametro o csv do usuario

    # Processar o CSV e obter o caminho do arquivo resultante
    try:
        resultado_file_path = executa(file_path)
        print(f"Arquivo resultante salvo em: {resultado_file_path}")
    except ValueError as e:
        print(f"Erro: {e}")