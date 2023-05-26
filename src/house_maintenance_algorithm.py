import datetime
import re


class MaintenanceInfo:
    def __init__(self, unom, address, coordinates, work_description, last_work_date):
        self.unom = unom
        self.address = address
        self.coordinates = coordinates
        self.work_description = work_description
        self.last_work_date = last_work_date


def parse_frequency(input_str):
    if input_str is None:
        return None
    
    input_str = input_str.lower()
    
    #год
    match = re.search(r'(\d+)\s+раз.*?год', input_str)
    if match:
        return 365 / int(match.group(1))
    
    match = re.search(r'раз\s+в\s+(\d+)\s+(год|лет)', input_str)
    if match:
        return 365 * int(match.group(1))

    #месяц
    match = re.search(r'(\d+)\s+раз.*?месяц', input_str)
    if match:
        return 30 / int(match.group(1))
    
    match = re.search(r'раз\s+в\s+(\d+)\s+месяц', input_str)
    if match:
        return 30 * int(match.group(1))

    #неделя
    match = re.search(r'(\d+)\s+раз.*?недел', input_str)
    if match:
        return 7 / int(match.group(1))
    
    match = re.search(r'раз\s+в\s+(\d+)\s+недел', input_str)
    if match:
        return 7 * int(match.group(1))

    #день
    match = re.search(r'(\d+)\s+раз.*?(день|сутк)', input_str)
    if match:
        return 1 / int(match.group(1))
    
    match = re.search(r'раз\s+в\s+(\d+)\s+(дн|сутк|суток)', input_str)
    if match:
        return 1 * int(match.group(1))
    
    if 'ежедневно' in input_str or 'раз в день' in input_str or 'раз(а) в день' in input_str:
        return 1

    if 'круглосуточно' in input_str:
        return 0

    return None


def order_maintenance_works_by_best_route(maintenance_works):
    #TODO
    return maintenance_works


def get_maintenance_works_by_date(db, date=None):
    current_date = datetime.date.today()
    if date is None:
        date = current_date
    if date < current_date:
        raise ValueError("The date cannot be earlier than the current day")

    maintenance_works = db.select_upcoming_maintenance_dates(date)
    maintenance_works = order_maintenance_works_by_best_route(maintenance_works)
    return maintenance_works
