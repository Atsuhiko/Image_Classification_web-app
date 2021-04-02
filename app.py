# 参考: https://qiita.com/redshoga/items/60db7285a573a5e87eb6
# 参考: https://teratail.com/questions/244325
# 参考: https://qiita.com/Susasan/items/52d1c838eb34133042a3
# DB参照：https://www.python.ambitious-engineer.com/archives/1640

import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, g, flash
from PIL import Image
import numpy as np
from datetime import datetime
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
# 判別アルゴリズムのインポート
from image_process import Predict_Woman_Man

app = Flask(__name__)

SAVE_DIR = "images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

@app.route('/images/<path:filepath>')
def send_js(filepath):
    return send_from_directory(SAVE_DIR, filepath)

@app.route("/", methods=["GET","POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":

        # 画像として読み込み
        image = request.files['image']
        input_img = Image.open(image)

        # 男と女の判別
        model = load_model('model.h5') # 学習ずみモデルの読み込み
        result = Predict_Woman_Man(input_img, model)
        #print(result)

        man = float(result[0][0]) # floar() しないとSQLにrealとして渡せない
        woman = float(result[0][1])

        if result[0][0] > result[0][1]:
            prediction = "man"
        else:
            prediction = "woman"

        # 判別後のラベルと時刻を付けてファイル名を付け、画像を保存
        filepath = prediction + datetime.now().strftime("_%Y%m%d%H%M%S") + ".jpg"
        save_path = os.path.join(SAVE_DIR, filepath)
        input_img.save(save_path)

        return  render_template("index.html", 
                                filepath=filepath, prediction=prediction, woman=woman, man=man)

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0', port=2222) # ポートの変更
