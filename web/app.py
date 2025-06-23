from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from werkzeug.utils import secure_filename

from hnsw.Recommander import Recommender

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'static/uploads'
DATA_FOLDER = '../archive/3/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 限制为5MB

Recommender = Recommender("../output/HNSW_60000_resnet34_heuristic_20_200", DATA_FOLDER)

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('show_results', filename=filename))

    return render_template('upload.html')


@app.route('/results/<filename>')
def show_results(filename):
    # print(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # print(filepath)
    start_time = time.time()
    similar_images = Recommender.recommend(filepath,k=8,ef_search=64)  # 8张相似图片
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时
    print(f"耗时: {elapsed_time:.6f} 秒")
    # print(similar_images)
    # print(similar_images)
    # similar_images = [filename]*8
    return render_template('results.html',
                           original=filename,
                           similar_images=similar_images)




@app.route('/data/<filename>')
def stored_file(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)