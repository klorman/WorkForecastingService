import psycopg2
import traceback
import configparser
import os
import pandas as pd
from .logs import Logs


class Database:
    def __init__(self):
        self.logs = Logs(__name__).get_logger()
        try:
            config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.ini')
            config = configparser.ConfigParser()
            config.read(config_file_path)

            # Установка соединения с базой данных
            self.conn = psycopg2.connect(
                host=config.get('postgresql', 'host'),
                database=config.get('postgresql', 'database'),
                user=config.get('postgresql', 'user'),
                password=config.get('postgresql', 'password')
            )

            # Создание объекта курсора
            self.cursor = self.conn.cursor()
            self.conn.autocommit = False
            self.logs.info("Connected to the database successfully")
        except (Exception, psycopg2.Error) as e:
            self.logs.error(f"Error occurred: {e}, Traceback: {traceback.format_exc()}")
            raise e


    def __del__(self):
        try:
            if self.conn:
                self.conn.commit()
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        except Exception as e:
            self.logs.error(f"Error occurred during the termination: {e}, Traceback: {traceback.format_exc()}")


    def insert_user_credentials(self, username, password_hash):
        try:
            query = "INSERT INTO user_credentials (username, password_hash) VALUES (%s, %s)"
            values = (username, password_hash)
            self.cursor.execute(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert user credentials: {e}, Traceback: {traceback.format_exc()}")


    def select_user_by_username(self, username):
        try:
            query = "SELECT id, username, password_hash FROM user_credentials WHERE username = %s"
            values = (username,)
            self.cursor.execute(query, values)
            return self.cursor.fetchone()
        except Exception as e:
            self.logs.error(f"Failed to fetch user by username: {e}, Traceback: {traceback.format_exc()}")


    def select_user_by_id(self, user_id):
        try:
            query = "SELECT id, username, password_hash FROM user_credentials WHERE id = %s"
            values = (user_id,)
            self.cursor.execute(query, values)
            return self.cursor.fetchone()
        except Exception as e:
            self.logs.error(f"Failed to fetch user by id: {e}, Traceback: {traceback.format_exc()}")


    def insert_resident_request(self, data):
        try:
            query = "INSERT INTO your_table (column1, column2, column3) VALUES (%s, %s, %s)"
            values = ("value1", "value2", "value3")
            self.cursor.execute(query, values)
            self.conn.commit()     
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert resident request: {e}, Traceback: {traceback.format_exc()}")


    def select_residents_requests(self):
        try:
            query = "SELECT * FROM your_table"
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=columns)
            return df
        except Exception as e:
            self.logs.error(f"Failed to fetch resident requests: {e}, Traceback: {traceback.format_exc()}")


    def insert_algorithm_result(self):
        try:
            query = "INSERT INTO your_table (column1, column2, column3) VALUES (%s, %s, %s)"
            values = ("value1", "value2", "value3")
            self.cursor.execute(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert algorithm result: {e}, Traceback: {traceback.format_exc()}")


    def select_algorithm_result_by_id(self, id):
        try:
            query = "SELECT * FROM your_table"
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=columns)
            return df
        except Exception as e:
            self.logs.error(f"Failed to fetch algorithm result by id: {e}, Traceback: {traceback.format_exc()}")
