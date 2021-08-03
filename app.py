'''
Desciption: Application.
'''
from flask import Flask, request
import json
from model import Classifier

# 初始化模型， 避免在函数内部初始化，耗时过长
bc = Classifier()
bc.load()

# 初始化flask
app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def gen_ans():
    '''
    @description: 以RESTful的方式获取模型结果, 传入参数为title: 图书标题， desc: 图书描述
    @param {type}
    @return: json格式， 其中包含标签和对应概率
    '''
    result = {}

    text = request.form['text']
    label = bc.predict(text)
    result = {
        "label": label
    }
    return json.dumps(result, ensure_ascii=False)


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
