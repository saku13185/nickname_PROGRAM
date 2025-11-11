from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# 仮ユーザーデータ
users = {
    "user1": "password1",
    "user2": "password2"
}

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username] == password:
            return redirect(url_for('index'))
        else:
            return render_template('result.html', message="ログイン失敗しました。")

    return redirect(url_for('home'))

@app.route('/index')
def index():
    return render_template('index.html')

# ✅ これを追加する！
@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    print("✅ 分析開始ボタンが押されました！")
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users:
            return render_template('result.html', message="すでに存在するユーザー名です。")

        users[username] = password
        return render_template('result.html', message="登録完了！ログインしてください。")

    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
