from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# ---------------------------------------------------
# サンプルのユーザー情報（簡易データベース）
# ---------------------------------------------------
users = {
    "user1": "password1",
    "user2": "password2"
}

# ---------------------------------------------------
# ルート: ログインページを表示
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template('login.html')


# ---------------------------------------------------
# ログイン処理
# ---------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # ユーザー認証チェック
        if username in users and users[username] == password:
            message = f"Welcome, {username}!"
            return render_template('result.html', message=message)
        else:
            message = "Login failed. Please check your username and password."
            return render_template('result.html', message=message)

    # GETメソッド（直接アクセス時）はログイン画面へ
    return redirect(url_for('home'))


# ---------------------------------------------------
# ユーザー登録処理
# ---------------------------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # すでに存在するユーザー名かチェック
        if username in users:
            message = "Username already exists. Please choose a different username."
            return render_template('result.html', message=message)

        # 新規登録
        users[username] = password
        message = "Registration successful! You can now log in."
        return render_template('result.html', message=message)

    # GETリクエスト時は登録フォーム表示
    return render_template('register.html')


# ---------------------------------------------------
# Flaskアプリ起動
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
