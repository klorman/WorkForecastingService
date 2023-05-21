from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import traceback

from database import Database
from logs import Logs

logs = Logs(__name__).get_logger()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

db = Database()
login_manager = LoginManager(app)


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
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error_login = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.find_by_username(username)

        if not user or not check_password_hash(user.password, password):
            error_login = 'Неправильное имя пользователя или пароль.'
            return render_template('login.html', error_message=error_login)

        login_user(user, remember=remember)
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error_signup = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.find_by_username(username)

        if user:
            error_signup = 'Пользователь с таким именем уже существует.'
        else:
            password_hash = generate_password_hash(password, method='sha256')
            db.insert_user_credentials(username, password_hash)
            return redirect(url_for('login'))

    return render_template('signup.html', error_message=error_signup)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


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


if __name__ == '__main__':
    app.run(debug=True)