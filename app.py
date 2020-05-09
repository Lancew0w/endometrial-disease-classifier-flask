from flask import Flask, render_template, request, redirect, url_for
import os
import static.KNN

app = Flask(__name__)


@app.route('/')
def index():
    # return 'Hello World!'
    return render_template('index.html')


@app.route('/upload_test', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static', 'test.csv')
        f.save(upload_path)
        try:
            static.KNN.test(os.path.join('static', 'test.csv'))
            return redirect(url_for('test_success'))
        except:
            return("加载数据集失败，请检查数据集格式问题。")
        return redirect(url_for('index'))
    return render_template('upload.html')


@app.route('/upload_train', methods=['POST', 'GET'])
def upload2():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static', 'train.csv')
        f.save(upload_path)
        try:
            static.KNN.train(os.path.join('static', 'train.csv'))
            return redirect(url_for('train_success'))
        except:
            return("使用训练集训练数据失败，请检查数据集格式。")
        return redirect(url_for('index'))
    return render_template('upload.html')


@app.route('/test_success')
def test_success():
    return render_template('test_success.html')

@app.route('/train_success')
def train_success():
    return render_template('train_success.html')


@app.route('/single_test', methods=['POST', 'GET'])
def single_test():
    if request.method == 'POST':
        fs = request.form['fs']
        dx = request.form['dx']
        cj = request.form['cj']
        xl = request.form['xl']
        zl = request.form['zl']
        hs = request.form['hs']
        nb = request.form['nb']
        fg = request.form['fg']
        bj = request.form['bj']

        try:
            args = [int(fs), float(dx), float(cj), int(xl), int(zl), int(hs), int(nb), int(fg), int(bj)]
            print(args)
            result = static.KNN.test_one(args)
            if result == 1:
                result_str = '(1)囊肿为恶行或孕期发生破裂扭转风险'
            if result == 2:
                result_str = '(2)囊肿观察共存至分娩风险'
            if result == 3:
                result_str = '(3)囊肿消失'
            return render_template('testone_success.html', result=result_str)

        except:
            return('填写数据有误，请后退重新检查后提交。')
        # return redirect(url_for('index'))
    return render_template('single_test.html')


if __name__ == '__main__':
    app.run()
