from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# サンプルのユーザー情報
users = {
    "user1": "password1",
    "user2": "password2"
}

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # ユーザー認証
    if username in users and users[username] == password:
        return render_template('result.html', message=f"Welcome, {username}!")
    else:
        return render_template('result.html', message="Login failed. Please check your username and password.")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # ユーザーが既に存在しないかチェック
        if username in users:
            return render_template('result.html', message="Username already exists. Please choose a different username.")

        # 新しいユーザーを登録
        users[username] = password
        return render_template('result.html', message="Registration successful! You can now log in.")

    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
