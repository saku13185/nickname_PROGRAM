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
            return render_template('result.html', message="로그인 실패했습니다.")

    return redirect(url_for('home'))

@app.route('/index')
def index():
    return render_template('index.html')

# ✅ これを追加する！
@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    print("✅ 분석 시작 버튼이 눌렸습니다!")
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users:
            return render_template('result.html', message="이미 존재하는 사용자 이름입니다.")

        users[username] = password
        return render_template('result.html', message="등록완료! 로그인해주세요.")

    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
