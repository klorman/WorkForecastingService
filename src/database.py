import psycopg2
import traceback
import configparser
import pandas as pd
from itertools import product
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


    def insert_house_mkd(self, rows):
        try:
            new_unoms = []
            for row in rows:
                try:
                    unom = int(row['COL_782']) if row['COL_782'] else None
                except ValueError:
                    continue

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
                          row['COL_3363'], row['COL_3468'], unom, coordinates)
                self.cursor.execute(query, values)
                self.conn.commit()

                status_message = self.cursor.statusmessage
                if "INSERT" in status_message:
                    new_unoms.append(row['COL_782'])

            if new_unoms == []:
                return
            work_types = self.get_all_maintenance_type_ids()
            if work_types == []:
                return
            rows = list(product(new_unoms, work_types))
            df = pd.DataFrame(rows, columns=['unom', 'work_type'])
            self.insert_into_maintenance_dates(df)
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert house mkd: {e}, Traceback: {traceback.format_exc()}")
            raise e

        
    def select_house_mkd(self):
        try:
            query = """
            SELECT house_mkd.col_756, col_758.name, house_mkd.col_760, house_mkd.col_761, house_mkd.col_762, house_mkd.col_763, 
            house_mkd.col_764, col_769.name, col_770.name, house_mkd.col_771, house_mkd.col_772, col_781.name, house_mkd.col_3363
            FROM public.house_mkd
            JOIN col_758 ON col_758.ID = house_mkd.col_758
            JOIN col_769 ON col_769.ID = house_mkd.col_769
            JOIN col_770 ON col_770.ID = house_mkd.col_770
            JOIN col_781 ON col_781.ID = house_mkd.col_781;
            """
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = ["COL_756", "COL_758", "COL_759", "COL_760", "COL_761", "COL_762", "COL_763", 
                       "COL_765", "COL_769", "COL_770", "COL_771", "COL_772", "COL_781", "COL_3363"]
            df = pd.DataFrame(data, columns=columns)
            df = df.fillna('')
            return df

        except Exception as e:
            self.logs.error(f"Failed to select house mkd: {e}, Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        

    def insert_incidents_urban(self, rows):
        try:
            if not rows:
                return
            
            rows = [row for row in rows if row['NAME'] is not None]

            # Проверяем, существуют ли комбинации name и source в таблице event_types
            event_types = {}
            for row in rows:
                try:
                    unom = int(row['UNOM']) if row['UNOM'] else None
                except ValueError:
                    continue

                event_type_key = (row['NAME'], row['SOURCE'])
                if event_type_key not in event_types:
                    query = "SELECT id FROM event_types WHERE name = %s AND source = %s"
                    values = event_type_key
                    self.cursor.execute(query, values)
                    result = self.cursor.fetchone()
                    if result:
                        event_types[event_type_key] = result[0]
                    else:
                        insert_query = "INSERT INTO event_types (name, source) VALUES (%s, %s) RETURNING id"
                        self.cursor.execute(insert_query, values)
                        event_types[event_type_key] = self.cursor.fetchone()[0]

            # Вставляем новые строки в incidents_urban с полученными event_type_id
            insert_query = """
                INSERT INTO incidents_urban (EVENT_TYPES, DATEBEGIN, DATEEND, DISTRICT, ADDRESS, UNOM)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = [(event_types[(row['NAME'], row['SOURCE'])], row['DATEBEGIN'], row['DATEEND'], row['DISTRICT'], row['ADDRESS'], unom) for row in rows]

            self.cursor.executemany(insert_query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert incident urban: {e}, Traceback: {traceback.format_exc()}")
            raise e
        

    def select_incidents_urban(self):
        try:
            query = """
            SELECT event_types.name, incidents_urban.dateend, incidents_urban.datebegin
            FROM public.incidents_urban
            JOIN incidents_urban ON incidents_urban.id = incidents_urban.event_types;
            """
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = ["Наименование", "Дата закрытия", "Дата создания во внешней системе"]
            df = pd.DataFrame(data, columns=columns)
            df = df.fillna('')
            return df

        except Exception as e:
            self.logs.error(f"Failed to select incidents urban: {e}, Traceback: {traceback.format_exc()}")
            return pd.DataFrame()


    def insert_major_repairs(self, row):
        try:
            try:
                unom = int(row['UNOM']) if row['UNOM'] else None
            except ValueError:
                return
            
            query = "SELECT id FROM major_repairs_types WHERE name = %s"
            values = (row['WORK_NAME'],)
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()
            if result is None:
                return
            major_rapair_type = result[0]

            query = """
            INSERT INTO major_repairs(PERIOD, WORK_NAME, NUM_ENTRANCE, ELEVATORNUMBER, PLAN_DATE_START, PLAN_DATE_END, 
            FACT_DATE_START, FACT_DATE_END, ADMAREA, DISTRICT, ADDRESS, UNOM) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (row['PERIOD'], major_rapair_type, row['NUM_ENTRANCE'], row['ElevatorNumber'], row['PLAN_DATE_START'], 
                      row['PLAN_DATE_END'], row['FACT_DATE_START'], row['FACT_DATE_END'], row['AdmArea'], 
                      row['District'], row['Address'], unom)
            self.cursor.execute(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert major repairs: {e}, Traceback: {traceback.format_exc()}")
            raise e
        

    def select_major_repairs(self):
        try:
            query = """
            SELECT major_repairs_types.name, major_repairs.plan_date_start, major_repairs.fact_date_start
            FROM public.major_repairs
            JOIN major_repairs_types ON major_repairs_types.id = major_repairs.work_name;
            """
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = ["WORK_NAME", "PLAN_DATE_START", "FACT_DATE_START"]
            df = pd.DataFrame(data, columns=columns)
            df = df.fillna('')
            return df

        except Exception as e:
            self.logs.error(f"Failed to select major repairs: {e}, Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        

    def get_training_data(self):
        df_house_mkd = self.select_house_mkd()
        df_incidents_urban = self.select_incidents_urban()
        df_major_repairs = self.select_major_repairs()
        df_combined = pd.concat([df_house_mkd, df_incidents_urban, df_major_repairs], axis=1)
        return df_combined
        

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
            values = (int(work_type),)
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            self.logs.error(f"Failed to fetch maintenance frequency: {e}, Traceback: {traceback.format_exc()}")
            return None
    

    def select_upcoming_maintenance_dates(self, date):
        try:
            query = """
            SELECT maintenance_types.name, house_mkd.name, maintenance_dates.next_work_date 
            FROM maintenance_dates
            JOIN maintenance_types ON maintenance_types.id = maintenance_dates.work_type
            JOIN house_mkd ON house_mkd.unom = maintenance_dates.unom
            WHERE next_work_date >= %s
            """
            values = (date,)
            self.cursor.execute(query, values)
            data = self.cursor.fetchall()
            columns = ["Работа", "Адрес", "Дата"]
            df = pd.DataFrame(data, columns=columns)
            df = df.fillna('')
            return df
        except Exception as e:
            self.logs.error(f"Failed to fetch upcoming maintenance dates: {e}, Traceback: {traceback.format_exc()}")
            return []


    def insert_into_maintenance_dates(self, rows):
        try:
            if rows is None:
                return

            query = """
                INSERT INTO maintenance_dates(unom, work_type, last_work_date, next_work_date) 
                VALUES (%s, %s, %s, %s);
            """
            values = []
            for _, row in rows.iterrows():
                unom = int(row['unom'])
                work_type = int(row['work_type'])
                maintenance_frequency = self.select_maintenance_frequency(work_type)
                if maintenance_frequency is None:
                    continue
                last_work_date = datetime.date.today()
                next_maintenance_date = last_work_date + datetime.timedelta(days=maintenance_frequency)
                values.append((int(unom), int(work_type), last_work_date, next_maintenance_date))
            self.cursor.executemany(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert into maintenance dates: {e}, Traceback: {traceback.format_exc()}")


    def insert_maintenance_types(self, rows):
        try:
            new_types = []
            for row in rows:
                try:
                    col_4489 = int(row['COL_4489']) if row['COL_4489'] else None
                except ValueError:
                    continue

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
                          row['COL_1239'], col_4489, row['COL_4923'])
                self.cursor.execute(query, values)
                self.conn.commit()

                status_message = self.cursor.statusmessage
                if "INSERT" in status_message:
                    inserted_id = self.cursor.fetchone()[0]
                    new_types.append(inserted_id)
                    
            if new_types == []:
                return
            unoms = self.get_all_unoms()
            if unoms == []:
                return
            combinations = list(product(unoms, new_types))
            df = pd.DataFrame(combinations, columns=['unom', 'work_type'])
            self.insert_into_maintenance_dates(df)          
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert maintenance types: {e}, Traceback: {traceback.format_exc()}")
            raise e


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


    def insert_major_repairs_results(self, rows):
        try:
            if rows is None:
                return

            query = "INSERT INTO major_repairs_results (major_repairs_types, unom, date) VALUES (%s, %s, %s)"
            values = []
            for row in rows:
                unom = row[0]
                major_rapairs_type = row[1]
                date = row[2]
                query = "SELECT id FROM major_repairs_types WHERE name = %s"
                values = (major_rapairs_type,)
                self.cursor.execute(query, values)
                work_type = self.cursor.fetchone()
                values.append((work_type, unom, date))
            self.cursor.executemany(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logs.error(f"Failed to insert algorithm result: {e}, Traceback: {traceback.format_exc()}")


    def select_major_repairs_results(self):
        try:
            query = """
            SELECT major_repairs_results.id, major_repairs_types.name, house_mkd.name, major_repairs_results.date
            FROM major_repairs_results
            JOIN major_repairs_types ON major_repairs_types.id = major_repairs_results.major_repairs_types
            JOIN house_mkd ON house_mkd.unom = major_repairs_results.unom
            """
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            columns = ["id", "Работа", "Адрес", "Дата"]
            df = pd.DataFrame(data, columns=columns)
            df = df.fillna('')
            return df
        except Exception as e:
            self.logs.error(f"Failed to fetch algorithm result by id: {e}, Traceback: {traceback.format_exc()}")


    def select_major_repairs_result_by_id(self, id):
        try:
            query = """
            SELECT major_repairs_types.name, major_repairs_results.unom, major_repairs_results.date
            FROM major_repairs_results
            JOIN major_repairs_types ON major_repairs_types.id = major_repairs_results.major_repairs_types
            WHERE id = %s
            """
            values = (id,)
            self.cursor.execute(query, values)
            data = self.cursor.fetchone()
            columns = ["Работа", "Адрес", "Дата"]
            df = pd.DataFrame(data, columns=columns)
            df = df.fillna('')
            return df
        except Exception as e:
            self.logs.error(f"Failed to fetch algorithm result by id: {e}, Traceback: {traceback.format_exc()}")


    def delete_row(self, table_name, id):
        try:
            query = f"DELETE FROM {table_name} WHERE id = %s"
            values = (id,)
            self.cursor.execute(query, values)
            self.connection.commit()
        except Exception as e:
            self.logs.error(f"Failed to delete row from {table_name}: {e}")
