from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import joblib
from classificaPeraComArvoreDecisao import executa as executa_arvore
from classificaPeraComKNN import executa as executa_knn

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/classifica', methods=['POST'])
def classifica_arvore():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
        
    if file:
        file_path = 'uploaded_file.csv'  # Salvar o arquivo temporariamente
        file.save(file_path)

        try:
            result_file_path = executa_arvore(file_path)

            # Read the processed data from the CSV file
            with open(result_file_path, 'r') as file:
                processed_data = pd.read_csv(file)

            # Convert the DataFrame to HTML table representation
            html_table = processed_data.to_html()

            return jsonify({"result": html_table})

        except ValueError as e:
            return jsonify({"error": str(e)}), 400

@app.route('/classifica_knn', methods=['POST'])
def classifica_knn():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        k = request.form.get('k')
        if not k:
            return "No K value provided", 400
        try:
            k = int(k)
        except ValueError:
            return "K value must be an integer", 400

        file_path = 'uploaded_file_knn.csv'  # Salvar o arquivo temporariamente
        file.save(file_path)

        try:
            result_file_path = executa_knn(file_path, k)

            # Ler os dados processados do arquivo CSV
            with open(result_file_path, 'r') as file:
                processed_data = pd.read_csv(file)

            # Converter o DataFrame para a representação de tabela HTML
            html_table = processed_data.to_html()

            return jsonify({"result": html_table})

        except ValueError as e:
            return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)