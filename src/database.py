import psycopg2
import traceback
import configparser
import pandas as pd
import datetime
import yandex_api
import time
from house_maintenance_algorithm import parse_frequency
from logs import Logs


class Database:
    def __init__(self):
        self.logs = Logs(__name__).get_logger()
        try:
            config = configparser.ConfigParser()
            config.read('configs/config.ini')
            time.sleep(10)
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


    def insert_house_mkd(self, row):
        try:
            query = """
            INSERT INTO house_mkd(NAME, COL_754, COL_756, COL_757, COL_758, COL_759, 
            COL_760, COL_761, COL_762, COL_763, COL_764, COL_769, COL_770, COL_771, COL_772, 
            COL_781, COL_2463, COL_3163, COL_3243, COL_3363, COL_3468, UNOM, COORDINATES) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (UNOM)
            DO UPDATE SET 
            NAME = EXCLUDED.NAME,
            COL_754 = EXCLUDED.COL_754, 
            COL_756 = EXCLUDED.COL_756, 
            COL_757 = EXCLUDED.COL_757,
            COL_758 = EXCLUDED.COL_758, 
            COL_759 = EXCLUDED.COL_759,
            COL_760 = EXCLUDED.COL_760,
            COL_761 = EXCLUDED.COL_761,
            COL_762 = EXCLUDED.COL_762,
            COL_763 = EXCLUDED.COL_763,
            COL_764 = EXCLUDED.COL_764,
            COL_769 = EXCLUDED.COL_769,
            COL_770 = EXCLUDED.COL_770,
            COL_771 = EXCLUDED.COL_771,
            COL_772 = EXCLUDED.COL_772,
            COL_781 = EXCLUDED.COL_781,
            COL_2463 = EXCLUDED.COL_2463,
            COL_3163 = EXCLUDED.COL_3163,
            COL_3243 = EXCLUDED.COL_3243,
            COL_3363 = EXCLUDED.COL_3363,
            COL_3468 = EXCLUDED.COL_3468,
            COORDINATES = EXCLUDED.COORDINATES;
            """
            coordinates = yandex_api.get_coordinates_by_address()
            values = (row['NAME'], row['COL_754'], row['COL_756'], row['COL_757'], 
                      row['COL_758'], row['COL_759'], row['COL_760'], row['COL_761'], row['COL_762'], 
                      row['COL_763'], row['COL_764'], row['COL_769'], row['COL_770'], row['COL_771'], 
                      row['COL_772'], row['COL_781'], row['COL_2463'], row['COL_3163'], row['COL_3243'], 
                      row['COL_3363'], row['COL_3468'], row['COL_782'], coordinates)
            self.cursor.execute(query, values)
            self.conn.commit()

            status_message = self.cursor.statusmessage
            if "INSERT" in status_message:
                work_types = self.get_all_maintenance_type_ids()
                for type in work_types:
                    self.insert_into_maintenance_dates(row['COL_782'], type)
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert house mkd: {e}, Traceback: {traceback.format_exc()}")

        
    def select_house_mkd(self):
        try:
            query = """
            SELECT house_mkd.id, house_mkd.name, house_mkd.col_754, house_mkd.col_756, house_mkd.col_757, col_758.name, 
            house_mkd.col_759, house_mkd.col_760, house_mkd.col_761, house_mkd.col_762, house_mkd.col_763, house_mkd.col_764, 
            col_769.name, col_770.name, house_mkd.col_771, house_mkd.col_772, col_781.name, house_mkd.unom, col_2463.name, 
            col_3163.name, col_3243.name, house_mkd.col_3363, house_mkd.col_3468, house_mkd.unom, house_mkd.coordinates
            FROM public.house_mkd
            JOIN col_758 ON col_758.ID = house_mkd.col_758
            JOIN col_769 ON col_769.ID = house_mkd.col_769
            JOIN col_770 ON col_770.ID = house_mkd.col_770
            JOIN col_781 ON col_781.ID = house_mkd.col_781
            JOIN col_2463 ON col_2463.ID = house_mkd.col_2463
            JOIN col_3163 ON col_3163.ID = house_mkd.col_3163
            JOIN col_3243 ON col_3243.ID = house_mkd.col_3243;
            """
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=columns)
            df = df.fillna('')
            return df

        except Exception as e:
            self.logs.error(f"Failed to select house mkd: {e}, Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        

    def insert_incidents_urban(self, row):
        try:
            if row['NAME'] is None:
                return
            
            # Проверяем, существует ли комбинация name и source в таблице event_types
            query = "SELECT id FROM event_types WHERE name = %s AND source = %s"
            values = (row['NAME'], row['SOURCE'])
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()

            if result:
                # Если комбинация name и source уже существует, получаем id из event_types
                event_type_id = result[0]
            else:
                # Если комбинация name и source не существует, добавляем новую строку в event_types
                insert_query = "INSERT INTO event_types (name, source) VALUES (%s, %s) RETURNING id"
                values = (row['NAME'], row['SOURCE'])
                self.cursor.execute(insert_query, values)
                event_type_id = self.cursor.fetchone()[0]

            # Вставляем новую строку в incidents_urban с полученным event_type_id
            query = """
            INSERT INTO incidents_urban (EVENT_TYPES, DATEBEGIN, DATEEND, DISTRICT, ADDRESS, UNOM)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = (event_type_id, row['DATEBEGIN'], row['DATEEND'], row['DISTRICT'], row['ADDRESS'], row['UNOM'])
            self.cursor.execute(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert incident urban: {e}, Traceback: {traceback.format_exc()}")


    def insert_major_repairs(self, row):
        try:
            query = """
            INSERT INTO incidents_urban(PERIOD, WORK_NAME, NUM_ENTRANCE, ELEVATORNUMBER, PLAN_DATE_START, PLAN_DATE_END, 
            FACT_DATE_START, FACT_DATE_END, ADMAREA, DISTRICT, ADDRESS, UNOM) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (PERIOD, WORK_NAME, NUM_ENTRANCE, ELEVATORNUMBER, PLAN_DATE_START, PLAN_DATE_END, 
            FACT_DATE_START, FACT_DATE_END, ADMAREA, DISTRICT, ADDRESS, UNOM)
            DO NOTHING;
            """
            values = (row['PERIOD'], row['WORK_NAME'], row['NUM_ENTRANCE'], row['ELEVATORNUMBER'], row['PLAN_DATE_START'], 
                      row['PLAN_DATE_END'], row['FACT_DATE_START'], row['FACT_DATE_END'], row['ADMAREA'], 
                      row['DISTRICT'], row['ADDRESS'], row['UNOM'])
            self.cursor.execute(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert major repairs: {e}, Traceback: {traceback.format_exc()}")
        

    def select_address_and_coordinates_by_unom(self, unom):
        try:
            query = "SELECT name, coordinates FROM house_mkd WHERE unom = %s"
            self.cursor.execute(query, (unom,))
            result = self.cursor.fetchone()
            if result:
                address = result[0]
                coordinates = result[1]
                return address, coordinates
            else:
                return None, None
        except Exception as e:
            self.logs.error(f"Failed to fetch name and coordinates for unom {unom}: {e}, Traceback: {traceback.format_exc()}")
            return None, None
    

    def get_all_unoms(self):
        try:
            query = "SELECT unom FROM house_mkd"
            self.cursor.execute(query)
            unoms = [item[0] for item in self.cursor.fetchall()]
            return unoms
        except Exception as e:
            self.logs.error(f"Failed to fetch unoms: {e}, Traceback: {traceback.format_exc()}")
            return []
        

    def get_all_maintenance_type_ids(self):
        try:
            query = "SELECT id FROM maintenance_types"
            self.cursor.execute(query)
            id = [item[0] for item in self.cursor.fetchall()]
            return id
        except Exception as e:
            self.logs.error(f"Failed to fetch work type ids: {e}, Traceback: {traceback.format_exc()}")
            return []
        

    def select_maintenance_frequency(self, work_type):
        try:
            query = """
            SELECT maintenance_periodicity.frequency
            FROM public.maintenance_types
            JOIN maintenance_periodicity ON maintenance_periodicity.ID = maintenance_types.maintenance_periodicity
            WHERE maintenance_types.id = %s"""
            values = (work_type,)
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            self.logs.error(f"Failed to fetch maintenance frequency: {e}, Traceback: {traceback.format_exc()}")
            return []
    

    def select_upcoming_maintenance_dates(self, date):
        try:
            query = "SELECT unom, work_type, last_work_date, next_work_date FROM maintenance_dates WHERE next_work_date >= %s"
            values = (date,)
            self.cursor.execute(query, values)
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            self.logs.error(f"Failed to fetch upcoming maintenance dates: {e}, Traceback: {traceback.format_exc()}")
            return []


    def insert_into_maintenance_dates(self, unom, work_type):
        try:
            query = """
            INSERT INTO maintenance_dates(unom, work_type, last_work_date, next_work_date) 
            VALUES (%s, %s, %s, %s);
            """
            maintenance_frequency = self.select_maintenance_frequency(work_type)
            last_work_date = None
            next_maintenance_date = None
            if maintenance_frequency:
                last_work_date = datetime.date.today()
                next_maintenance_date = last_work_date + datetime.timedelta(days=maintenance_frequency)

            values = (unom, work_type, last_work_date, next_maintenance_date)
            self.cursor.execute(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert into maintenance dates: {e}, Traceback: {traceback.format_exc()}")


    def insert_maintenance_types(self, row):
        try:
            query = "SELECT id FROM maintenance_periodicity WHERE work_description = %s"
            values = (row['COL_1044'],)
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()

            periodicity_id = None
            if result:
                periodicity_id = result[0]
            elif row['COL_1044'] is not None:
                maintenance_periodicity = parse_frequency(row['COL_1044'])
                insert_query = "INSERT INTO maintenance_periodicity (work_description, frequency) VALUES (%s, %s) RETURNING id"
                values = (row['COL_1044'], maintenance_periodicity)
                self.cursor.execute(insert_query, values)
                periodicity_id = self.cursor.fetchone()[0]

            query = """
            INSERT INTO maintenance_types(NAME, maintenance_periodicity, COL_1157, COL_1239, COL_4489, COL_4923) 
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (NAME)
            DO UPDATE SET 
            maintenance_periodicity = EXCLUDED.maintenance_periodicity, 
            COL_1157 = EXCLUDED.COL_1157, 
            COL_1239 = EXCLUDED.COL_1239,
            COL_4489 = EXCLUDED.COL_4489, 
            COL_4923 = EXCLUDED.COL_4923
            RETURNING id;
            """
            values = (row['NAME'], periodicity_id, row['COL_1157'], 
                      row['COL_1239'], row['COL_4489'], row['COL_4923'])
            self.cursor.execute(query, values)
            self.conn.commit()

            status_message = self.cursor.statusmessage
            if "INSERT" in status_message:
                inserted_id = self.cursor.fetchone()[0]
                unoms = self.get_all_unoms()
                for unom in unoms:
                    self.insert_into_maintenance_dates(unom, inserted_id)
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert maintenance types: {e}, Traceback: {traceback.format_exc()}")


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


    def insert_algorithm_result(self):
        try:
            #TODO
            query = "INSERT INTO your_table (column1, column2, column3) VALUES (%s, %s, %s)"
            values = ("value1", "value2", "value3")
            self.cursor.execute(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert algorithm result: {e}, Traceback: {traceback.format_exc()}")


    def select_algorithm_result_by_id(self, id):
        try:
            #TODO
            query = "SELECT * FROM major_repairs_results"
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=columns)
            return df
        except Exception as e:
            self.logs.error(f"Failed to fetch algorithm result by id: {e}, Traceback: {traceback.format_exc()}")
