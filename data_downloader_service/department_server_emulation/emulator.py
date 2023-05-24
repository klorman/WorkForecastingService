from flask import Flask, jsonify, send_file
import os
import configparser

app = Flask(__name__)

config_file_path = os.path.join(os.path.dirname(__file__), 'configs', 'emulator.config')
config = configparser.ConfigParser()
config.read(config_file_path)
folder_path = config.get('emulator', 'folder_path')

@app.route('/files', methods=['GET'])
def get_file_list():
    files = os.listdir(folder_path)
    return jsonify(files)

@app.route('/files/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(folder_path, filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)