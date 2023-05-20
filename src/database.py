import psycopg2
import pandas as pd


class Database:
    def __init__(self):
        # Установка соединения с базой данных
        self.conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="WorkForecastingService",
            user="postgre",
            password="VzeVzeVze"
        )

        # Создание объекта курсора
        self.cursor = self.conn.cursor()


    def __del__(self):
        self.cursor.close()
        self.conn.close()


    def insert_user_credentials(self, login, password_hash):
        # SQL-запрос для добавления данных аутентификации
        query = "INSERT INTO your_table (login, password_hash) VALUES (%s, %s)"
        values = (login, password_hash)

        # Выполнение SQL-запроса с передачей значений
        self.cursor.execute(query, values)

        # Фиксация изменений
        self.conn.commit()      


    def select_password_hash_by_login(self, login):
        # Выполнение SQL-запроса
        query = "SELECT password_hash FROM your_table WHERE login = %s"
        values = (login,)
        self.cursor.execute(query, values)

        # Получение данных из запроса
        password_hash = self.cursor.fetchone()[0]

        return password_hash


    def insert_resident_request(self, data):
        # SQL-запрос для добавления строки
        query = "INSERT INTO your_table (column1, column2, column3) VALUES (%s, %s, %s)"

        # Значения для вставки в столбцы
        values = ("value1", "value2", "value3")

        # Выполнение SQL-запроса с передачей значений
        self.cursor.execute(query, values)

        # Фиксация изменений
        self.conn.commit()      


    def select_residents_requests(self):
        # Выполнение SQL-запроса
        query = "SELECT * FROM your_table"
        self.cursor.execute(query)

        # Получение данных из результата запроса
        data = self.cursor.fetchall()

        # Получение названий столбцов
        columns = [desc[0] for desc in self.cursor.description]

        # Создание таблицы Pandas из данных
        df = pd.DataFrame(data, columns=columns)

        return df


    def insert_algorithm_result(self):
        # SQL-запрос для добавления строки
        query = "INSERT INTO your_table (column1, column2, column3) VALUES (%s, %s, %s)"

        # Значения для вставки в столбцы
        values = ("value1", "value2", "value3")

        # Выполнение SQL-запроса с передачей значений
        self.cursor.execute(query, values)

        # Фиксация изменений
        self.conn.commit()     


    def select_algorithm_result_by_id(self, id):
        # Выполнение SQL-запроса
        query = "SELECT * FROM your_table"
        self.cursor.execute(query)

        # Получение данных из результата запроса
        data = self.cursor.fetchall()

        # Получение названий столбцов
        columns = [desc[0] for desc in self.cursor.description]

        # Создание таблицы Pandas из данных
        df = pd.DataFrame(data, columns=columns)

        return df
