from flask import Flask, render_template, request
from ml import main
import pandas as pd
app = Flask(__name__)


@app.route('/')
def output():
    #return render_template('mlOutput.html',title='flask test',accuracy=test.ml()) #渡す引数
    return render_template("firstPage.html")

#form actionに反応する
@app.route("/page1", methods=["GET"])
def page1():
    return render_template("page1.html")

@app.route("/page2", methods=["GET"])
def page2():
    return render_template("page2.html")

@app.route("/dl", methods=["GET"])
def dl():
    accuracy=main.ml()
    return render_template('mlOutput.html',title='flask test',accuracy=accuracy[1])

@app.route("/ml", methods=["POST"])
def ml():
    accuracy=main.titanic_original(main.csv_load(request.form.get('data')))
    return render_template('mlOutput.html',title='flask test',accuracy=accuracy)

@app.route("/load_data", methods=["POST"])
def load_data():
    #columns_name = main.conv_data(main.serch_null(main.csv_load(request.form.get('data'))))[0]
    #columns_type = main.conv_data(main.serch_null(main.csv_load(request.form.get('data'))))[1]
    #return render_template('load_data.html',title='flask test',columns_name=columns_name,columns_type=columns_type)
    file = request.files['data']
    file.save('data.csv')
    columns_nulldata = main.serch_null(main.csv_load(request.files['data']))[0]#nullのあるカラムと個数を返す
    columns_name = main.serch_null(main.csv_load(request.files['data']))[1]
    return render_template('load_data.html',title='flask test',columns_name=columns_name,columns_nulldata=columns_nulldata)

@app.route("/null_conv", methods=["POST"])
def null_conv():
    columns_nulldata = main.receive_data()[2]
    columns_name = main.receive_data()[1]
    radio_data = []
    drop_columns = {}
    target = ""
    requirements_data = {}


    
    for i in columns_nulldata:
        radio_data.append(request.form.get(i))

    for name in columns_name:
        drop_columns[name] =request.form.get(name + "_drop") 

    for name in columns_name:
        requirements_data[name] = "no_change"

    for i in range(len(columns_nulldata)):
        requirements_data[columns_nulldata[i]] = radio_data[i]

    for name in columns_name:
        if drop_columns[name] == "drop":
            requirements_data[name] = drop_columns[name]
    target = request.form.get("target")
    model = request.form.get('model')
    test = []
    for i in range(5):
        test.append(request.form.get(str(i + 1))); 
    accuracy = main.titanic(requirements_data,target,model)
    return render_template('null_conv.html',title='flask test',radio=radio_data,target=target,accuracy=accuracy,test=test)

@app.route("/load", methods=["GET"])
def load():
    return render_template('loading.html',title='flask test')
    ml()

@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

@app.route("/login_manager", methods=["POST"])  #追加
def login_manager():
    return "ようこそ、" + request.form.get('select') + "さん"


#おまじない
if __name__ == "__main__":
    app.run(debug=True)