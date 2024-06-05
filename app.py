from flask import Flask, request, redirect, url_for, send_file
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
from mainKNN import load_knn_model
from sklearn.preprocessing import LabelEncoder  # Importação necessária

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Carregar o modelo de árvore de decisão treinado
modelo_arvore = joblib.load('modelos/modelo_arvore_decisao.pkl')

# Carregar o modelo KNN e normalizador
knn_model_path = 'modelos/modelo_knn.pkl'
knn_scaler_path = 'modelos/normalizador_knn.pkl'

# Rota para a página de upload
@app.route('/')
def upload_file():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Upload CSV</title>
    </head>
    <body>
        <h1>Upload CSV</h1>
        <form action="/uploader" method="post" enctype="multipart/form-data">
            <label for="file">Escolha o arquivo CSV:</label>
            <input type="file" name="file" id="file">
            <br><br>
            <label for="model_choice">Escolha o modelo:</label>
            <select name="model_choice" id="model_choice" onchange="toggleKValue()">
                <option value="arvore">Árvore de Decisão</option>
                <option value="knn">KNN</option>
            </select>
            <br><br>
            <div id="k_value_div" style="display:none;">
                <label for="k_value">Escolha o valor de k para o KNN:</label>
                <input type="number" name="k_value" id="k_value" value="3" min="1">
                <br><br>
            </div>
            <input type="submit" value="Upload">
        </form>

        <script>
            function toggleKValue() {
                var modelChoice = document.getElementById('model_choice').value;
                var kValueDiv = document.getElementById('k_value_div');
                if (modelChoice === 'knn') {
                    kValueDiv.style.display = 'block';
                } else {
                    kValueDiv.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    '''

# Rota para processar o arquivo CSV e calcular previsões
@app.route('/uploader', methods=['POST'])
def uploader_file():
    if 'file' not in request.files or 'model_choice' not in request.form:
        return redirect(request.url)
    
    file = request.files['file']
    model_choice = request.form['model_choice']
    k_value = request.form.get('k_value', None)
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Processar o arquivo
        dados = pd.read_csv(file_path)
        
        if 'qualidade' in dados.columns:
            return "O arquivo CSV não deve conter a coluna 'qualidade'.", 400
        
        x = dados
        
        if model_choice == 'knn':
            if k_value is None:
                return "Por favor, forneça o valor de k para o modelo KNN.", 400
            
            k_value = int(k_value)
            
            # Carregar e ajustar o modelo KNN com o valor de k fornecido
            modelo_knn, normalizacao = load_knn_model(knn_model_path, knn_scaler_path)
            modelo_knn.n_neighbors = k_value
            
            # Normalizar os dados de entrada
            x_normalizado = normalizacao.transform(x)
            y_pred_knn = modelo_knn.predict(x_normalizado)
            
            # Converter previsões para rótulos 'boa' ou 'ruim'
            label_encoder = LabelEncoder()
            label_encoder.fit(['ruim', 'boa'])
            y_pred_knn_label = label_encoder.inverse_transform(y_pred_knn)
            
            # Adicionar previsões ao DataFrame
            dados['qualidade'] = y_pred_knn_label
        else:
            # Fazer previsões com o modelo de árvore de decisão
            y_pred_arvore = modelo_arvore.predict(pd.DataFrame(x, columns=x.columns))
            
            # Converter previsões para rótulos 'boa' ou 'ruim'
            label_encoder = LabelEncoder()
            label_encoder.fit(['ruim', 'boa'])
            y_pred_arvore_label = label_encoder.inverse_transform(y_pred_arvore)
            
            # Adicionar previsões ao DataFrame
            dados['qualidade'] = y_pred_arvore_label
        
        # Salvar o DataFrame com as previsões como CSV
        result_filename = 'result_' + filename
        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        dados.to_csv(result_file_path, index=False)
        
        return send_file(result_file_path, as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
