from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import traceback
import os
import configparser
import io
from house_maintenance_algorithm import get_maintenance_works_by_date
from database import Database
from logs import Logs
from import_data import import_data
import pandas as pd
from model import get_result, additional_training, get_model_and_opt


logs = Logs(__name__).get_logger()
app = Flask(__name__)
config = configparser.ConfigParser()
config.read('configs/app.config')
app.config['SECRET_KEY'] = config.get('app', 'secret_key')
app.config['UPLOAD_FOLDER'] = config.get('app', 'upload_folder')

model, opt = get_model_and_opt()
db = Database()
login_manager = LoginManager(app)
major_repairs_results_df = db.select_major_repairs_results()
maintenance_works_df = get_maintenance_works_by_date(db)


def get_results(model):
    try:
        df_house_mkd = db.select_house_mkd()
        df_incidents_urban = db.select_incidents_urban()
        df_major_repairs = db.select_major_repairs()

        model_result = get_result(df_house_mkd, df_incidents_urban, df_major_repairs, model)
        print(model_result)
        unom = "unom_value"
        date = "date_value"
        rows = []
        for work_id, preds in model_result.items():
            major_rapairs_type = preds
            row = (unom, major_rapairs_type, date)
            print(row)
            rows.append(row)
        db.insert_major_repairs_results(rows)
        major_repairs_results_df = db.select_major_repairs_results()
        print("Data inserted successfully.")
    except Exception as e:
        print("An error occurred:", e)


class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    @staticmethod
    def get(user_id):
        user_data = db.select_user_by_id(user_id)
        if user_data:
            return User(*user_data)

    @staticmethod
    def find_by_username(username):
        user_data = db.select_user_by_username(username)
        if user_data:
            return User(*user_data)


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@app.errorhandler(Exception)
def handle_error(e):
    logs.error(f"An error occurred: {str(e)}, Traceback: {traceback.format_exc()}")
    return {"message": "An error occurred, please try again later."}, 500


@app.route('/')
def index():
    return redirect(url_for('login_signup'))


@app.route('/submit_changes', methods=['POST'])
@login_required
def submit_changes():
    data = request.get_json()
    ids_to_delete = data['ids_to_delete']
    rows_to_update = data['rows_to_update']
    try:
        for id in ids_to_delete:
            db.delete_row('major_repairs_results', ids_to_delete)
        for row in rows_to_update:
            id = row['id']
            date = row['date']
            db.update_major_repairs_result_by_id(date, id)
        return jsonify(status='success')
    except Exception as e:
        return jsonify(status='failure', message=str(e))
    

@app.route('/login', methods=['GET', 'POST'])
def login_signup():
    error_login = ''
    error_signup = ''

    if request.method == 'POST':
        if 'login' in request.form:
            username = request.form.get('username')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False

            user = User.find_by_username(username)

            if not user or not check_password_hash(user.password, password):
                error_login = 'Неправильное имя пользователя или пароль.'
            else:
                login_user(user, remember=remember)
                return redirect(url_for('dashboard'))
        elif 'signup' in request.form:
            username = request.form.get('username')
            password = request.form.get('password')

            user = User.find_by_username(username)

            if user:
                error_signup = 'Пользователь с таким именем уже существует.'
            else:
                password_hash = generate_password_hash(password, method='sha256')
                db.insert_user_credentials(username, password_hash)
                return redirect(url_for('login_signup'))

    return render_template('site.html', error_login=error_login, error_signup=error_signup)


@app.route('/load_data', methods=['GET'])
@login_required
def load_data():
    page = request.args.get('page', default = 1, type = int)
    page_size = request.args.get('page_size', default = 100, type = int)
    table = request.args.get("table")

    start = (page - 1) * page_size
    end = start + page_size
    if table == "major_repairs_results":
        major_repairs_table = major_repairs_results_df.iloc[start:end].to_dict("records")
        return jsonify({"table1": major_repairs_table})
    elif table == "maintenance_works_results":
        maintenance_table = maintenance_works_df.iloc[start:end].to_dict("records")
        return jsonify({"table2": maintenance_table})
    else:
        return jsonify({})


@app.route('/dashboard')
@login_required
def dashboard():
    major_repairs_table = major_repairs_results_df.head(100).to_dict('records')
    maintenance_table = maintenance_works_df.head(100).to_dict('records')
    return render_template('alg.html', table1=major_repairs_table, table2=maintenance_table)


@app.route('/algorithm', methods=['POST'])
@login_required
def algorithm():
    flash('Алгоритм в процессе разработки', 'info')
    return redirect(url_for('dashboard'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/download', methods=['GET'])
@login_required
def download():
    df = db.select_major_repairs_results()
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        data = output.getvalue()
    return Response(
        data,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-disposition":
                 "attachment; filename=major_repairs_results.xlsx"})


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result = import_data(db, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if result == 0:
            df_house_mkd = db.select_house_mkd()
            df_incidents_urban = db.select_incidents_urban()
            df_major_repairs = db.select_major_repairs()
            model, opt = additional_training(df_house_mkd, df_incidents_urban, df_major_repairs, opt, model)
            flash('File successfully uploaded')
        else:
            flash('An error occurred while uploading the file')
        return redirect(url_for('dashboard'))


if __name__ == '__main__':
    get_results(model)
    app.run(host='0.0.0.0', port=5000, debug=False)
