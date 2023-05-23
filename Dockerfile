# Используем базовый образ Ubuntu 22.04
FROM ubuntu:22.04

# Устанавливаем необходимые переменные среды
ENV DEBIAN_FRONTEND=noninteractive

# Установка необходимых пакетов для Ubuntu и Python
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    postgresql \ 
    libpq-dev \
    python3-dev

# Установка зависимостей Python
COPY requirements.txt /app/requirements.txt
COPY work.backup /app/work.backup
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Копирование исходного кода приложения в контейнер
COPY /src/. /app

#Настройка postgresql
RUN service postgresql start \
    && su - postgres -c "psql -c \"ALTER USER postgres WITH PASSWORD 'VzeVzeVze'\"" \
    && su - postgres -c "psql -c \"CREATE DATABASE workforecastingservice;\"" \
    && su - postgres -c "pg_restore -U postgres -d workforecastingservice /app/work.backup"

# Запуск БД и Flask-приложения
CMD service postgresql start && cd app && python3 app.py