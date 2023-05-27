import pandas as pd
import numpy as np
from datetime import datetime


house_mkd_columns = ['NAME', 'COL_754', 'COL_756', 'COL_757', 'COL_758', 'COL_759', 'COL_760', 'COL_761', 'COL_762', 
                     'COL_763', 'COL_764', 'COL_769', 'COL_770', 'COL_771', 'COL_772', 'COL_781', 'COL_782', 'COL_2463', 
                     'COL_3163', 'COL_3243', 'COL_3363', 'COL_3468']

incidents_urban_columns = ['Наименование', 'Источник', 'Дата создания во внешней системе', 'Дата закрытия', 'Округ', 'Адрес', 'unom']

major_repairs_columns = ['PERIOD', 'WORK_NAME', 'NUM_ENTRANCE', 'ElevatorNumber', 'PLAN_DATE_START', 'PLAN_DATE_END', 
                         'FACT_DATE_START', 'FACT_DATE_END', 'AdmArea', 'District', 'Address', 'UNOM']

maintenance_data_columns = ['NAME', 'COL_1044', 'COL_1157', 'COL_1239', 'COL_4489', 'COL_4923']


def import_data(db, file_path):
    df = pd.read_excel(file_path)
    columns = df.columns.tolist()
    
    column_sets = {
        'house_mkd': set(house_mkd_columns),
        'incidents_urban': set(incidents_urban_columns),
        'major_repairs': set(major_repairs_columns),
        'maintenance_data': set(maintenance_data_columns)
    }

    matched_table = None
    for table, columns_set in column_sets.items():
        if columns_set.issubset(columns):
            matched_table = table
            break

    if matched_table is None:
        return 1
    
    try:
        df = df.replace({np.nan: None})
        if matched_table == 'house_mkd':
            import_house_mkd(db, df)
        elif matched_table == 'incidents_urban':
            import_incidents_urban(db, df)
        elif matched_table == 'major_repairs':
            import_major_repairs(db, df)
        elif matched_table == 'maintenance_data':
            import_maintenance_types(db, df)
    except:
        return 1
    
    return 0
    

def import_house_mkd(db, df):
    df = df[['NAME', 'COL_754', 'COL_756', 'COL_757', 'COL_758', 'COL_759', 'COL_760', 
             'COL_761', 'COL_762', 'COL_763', 'COL_764', 'COL_769', 'COL_770', 'COL_771', 'COL_772', 
             'COL_781', 'COL_782', 'COL_2463', 'COL_3163', 'COL_3243', 'COL_3363', 'COL_3468']]

    rows = df.to_dict('records')
    db.insert_house_mkd(rows)


def import_incidents_urban(db, df):
    df = df.rename(columns={'Наименование': 'NAME', 'Источник': 'SOURCE', 'Дата создания во внешней системе': 'DATEBEGIN', 
                            'Дата закрытия': 'DATEEND', 'Округ': 'DISTRICT', 'Адрес': 'ADDRESS', 'unom': 'UNOM'})
    rows = df.to_dict('records')
    db.insert_incidents_urban(rows)


def import_major_repairs(db, df):
    df = df[['PERIOD', 'WORK_NAME', 'NUM_ENTRANCE', 'ElevatorNumber', 'PLAN_DATE_START', 'PLAN_DATE_END', 
             'FACT_DATE_START', 'FACT_DATE_END', 'AdmArea', 'District', 'Address', 'UNOM']]

    for _, row in df.iterrows():
        db.insert_major_repairs(row)


def import_maintenance_types(db, df):
    df = df[['NAME', 'COL_1044', 'COL_1157', 'COL_1239', 'COL_4489', 'COL_4923']]
    df = df[df['COL_4923'] >= datetime.now().year]

    rows = df.to_dict('records')
    db.insert_maintenance_types(rows)
