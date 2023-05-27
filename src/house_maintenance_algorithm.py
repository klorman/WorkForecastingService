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


def schedule_tasks(new_tasks, scheduled_tasks):
    start_date = datetime.date.today()
    end_date = start_date + datetime.timedelta(days=1)

    # Create list of all dates in the next year
    all_dates = [start_date + datetime.timedelta(days=i) for i in range(365)]

    # Create a list of dates when tasks are already scheduled
    scheduled_dates = []
    for last_date, frequency in scheduled_tasks:
        next_date = last_date + datetime.timedelta(days=frequency)
        while next_date <= end_date:
            scheduled_dates.append(next_date)
            next_date += datetime.timedelta(days=frequency)

    # Sort new_tasks by frequency, higher frequency tasks will be scheduled first
    sorted_new_tasks = sorted([(freq, idx) for idx, freq in enumerate(new_tasks)], reverse=True)
    task_schedule = []

    for frequency, task_id in sorted_new_tasks:
        for date in all_dates:
            # Ignore dates when tasks are already scheduled
            if date in scheduled_dates:
                continue
            if date.toordinal() % frequency == 0:
                task_schedule.append((task_id, date))
                break

    return task_schedule

if __name__ == "__main__":
    # Use function
    new_tasks = [5, 7, 10, 1, 1]
    scheduled_tasks = [(datetime.date.today() - datetime.timedelta(days=5), 7),
                       (datetime.date.today() - datetime.timedelta(days=10), 10),
                       (datetime.date.today() - datetime.timedelta(days=11), 10)]
    schedule = schedule_tasks(new_tasks, scheduled_tasks)

    for task_id, date in schedule:
        print(f'Task ID {task_id}: {date}')


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
