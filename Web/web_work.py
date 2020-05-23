# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import sys
from flask import make_response
from flask import send_from_directory
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_uploads import UploadSet, configure_uploads, patch_request_class

# Some utilites
import numpy as np
import pandas as pd
from DataProcess import DataProcess
from Inference import Inference

# Declare a flask app
app = Flask(__name__)

app.config['UPLOADED_DATAS_DEST'] = os.getcwd()  # 文件储存地址

datas = UploadSet('datas')
configure_uploads(app, datas)
patch_request_class(app)  # 文件大小限制，默认为16MB

#模型变量声明----------------------------------1


@app.route('/get_attachment')
def get_attachment(filename):

    file = request.files.get('web_work.ipnb')  # 获取文件
    filename = file.filename  # 获取文件名
    file.save(os.path.join(FILE_DIR, filename))  # 保存文件
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,
                               as_attachment=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    if request.method == 'POST' and 'data' in request.files:
        for filename in request.files.getlist('data'):
            datas.save(filename)
        dataset = DataProcess(os.getcwd() + "/base_test_sum.csv",
                              os.getcwd() + "/knowledge_test_sum.csv",
                              os.getcwd() + "/money_report_test_sum.csv",
                              os.getcwd() + "/year_report_test_sum.csv")
        #print(Inference(dataset))
        Inference(dataset).to_csv(os.getcwd() + "/result.csv",
                                  encoding="utf-8",
                                  index=False)

    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #预测、得到csv文件、生成predict.csv存在os.getcwd()------------------------------------3

    return send_from_directory(os.getcwd(), "result.csv", as_attachment=True)


if __name__ == '__main__':
    app.run(port=5004, threaded=False)
