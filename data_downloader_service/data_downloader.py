import os
import requests
import configparser
import time
from logs import Logs

logs = Logs(__name__).get_logger()

class FileDownloader:
    def __init__(self, server_url, local_dir, file_extension):
        self.server_url = server_url
        self.local_dir = local_dir
        self.file_extension = file_extension

    def download_files(self):
        response = requests.get(self.server_url)
        if response.status_code == 200:
            files = response.json()
            for file in files:
                if file.endswith(self.file_extension):
                    file_url = f"{self.server_url}/{file}"
                    file_path = os.path.join(self.local_dir, file)
                    self.download_file(file_url, file_path)
                    logs.info(f"File {file} downloaded successfully.")
        else:
            logs.error("Error accessing the server.")

    def download_file(self, url, file_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
        else:
            logs.error(f"Error downloading file from {url}.")

    def run(self, time_period):
        while True:
            self.download_files()
            time.sleep(time_period)

if __name__ == '__main__':
    config_file_path = os.path.join(os.path.dirname(__file__), 'configs', 'data_downloader.config')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    server_url = config.get('downloader', 'server_url')
    local_dir = config.get('downloader', 'local_dir')
    file_extension = config.get('downloader', 'file_extension')
    time_period = int(config.get('downloader', 'time_period'))

    downloader = FileDownloader(server_url, local_dir, file_extension)
    downloader.run(time_period)