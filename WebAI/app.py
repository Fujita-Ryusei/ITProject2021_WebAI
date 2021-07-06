from flask import Flask, render_template, request
from ml import main
import pandas as pd
from PIL import Image
app = Flask(__name__)


#def reload_selection():
#    data = pd.read_csv('data.csv')
#    columns_nulldata = main.serch_null(data)[0]#nullのあるカラムと個数を返す
#    columns_name = main.serch_null(data)[1]
#    return render_template('selection_page.html',columns_name=columns_name,columns_nulldata=columns_nulldata)

@app.route('/')
def output():
    #return render_template('mlOutput.html',title='flask test',accuracy=test.ml()) #渡す引数
    return render_template("front_page.html")

@app.route("/selection_page", methods=["POST"])
def load_data():
    #columns_name = main.conv_data(main.serch_null(main.csv_load(request.form.get('data'))))[0]
    #columns_type = main.conv_data(main.serch_null(main.csv_load(request.form.get('data'))))[1]
    #return render_template('load_data.html',title='flask test',columns_name=columns_name,columns_type=columns_type)
    try:

        #img = Image.open("test.jpg")

        file = request.files['data']
        file.save('data.csv')
        columns_nulldata = main.serch_null(main.csv_load(request.files['data']))[0]#nullのあるカラムと個数を返す
        columns_name = main.serch_null(main.csv_load(request.files['data']))[1]
        #columns_nulldata = main.receive_data()[2]
        #columns_name = main.receive_data()[1]
        return render_template('selection_page.html',columns_name=columns_name,columns_nulldata=columns_nulldata)
    except Exception as e:
        return render_template("front_page.html")

@app.route("/result_page", methods=["POST"])
def null_conv():
    try:
        columns_nulldata = main.receive_data()[2]
        columns_name = main.receive_data()[1]
        radio_data = []
        drop_columns = {}
        target = ""
        requirements_data = {}
        e =""


        target = request.form.get("target")
        model = request.form.get('model')
        ############
        if not target and not model:
            data = pd.read_csv('data.csv')
            columns_nulldata = main.serch_null(data)[0]#nullのあるカラムと個数を返す
            columns_name = main.serch_null(data)[1]
            e = "目的変数とモデルが選択されていません"
            return render_template('selection_page.html',columns_name=columns_name,columns_nulldata=columns_nulldata,detail_error=e)
        elif not target:
            data = pd.read_csv('data.csv')
            columns_nulldata = main.serch_null(data)[0]#nullのあるカラムと個数を返す
            columns_name = main.serch_null(data)[1]
            e = "目的変数が選択されていません"
            return render_template('selection_page.html',columns_name=columns_name,columns_nulldata=columns_nulldata,detail_error=e)            
        elif not model:
            data = pd.read_csv('data.csv')
            columns_nulldata = main.serch_null(data)[0]#nullのあるカラムと個数を返す
            columns_name = main.serch_null(data)[1]
            e = "モデルが選択されていません"
            return render_template('selection_page.html',columns_name=columns_name,columns_nulldata=columns_nulldata,detail_error=e)
        
        for i in columns_nulldata:
            if request.form.get(i) != None:
                radio_data.append(request.form.get(i))
            else:
                data = pd.read_csv('data.csv')
                columns_nulldata = main.serch_null(data)[0]#nullのあるカラムと個数を返す
                columns_name = main.serch_null(data)[1]
                e = "欠損値が選択されていません"
                return render_template('selection_page.html',columns_name=columns_name,columns_nulldata=columns_nulldata,detail_error=e)

        
        ############################
        # 欠損があるカラムがあるのにradiodataに値がない
        #if  len(columns_nulldata) != 0 and len(columns_nulldata) != len(radio_data):

        for name in columns_name:
            drop_columns[name] =request.form.get(name + "_drop") 

        for name in columns_name:
            requirements_data[name] = "no_change"

        for i in range(len(columns_nulldata)):
            requirements_data[columns_nulldata[i]] = radio_data[i]

        for name in columns_name:
            if drop_columns[name] == "drop":
                requirements_data[name] = drop_columns[name]

        #完成版ではモデルによってパラメータの数が異なる
        #modelによってforを分岐させるとかする関数を作る
        #Noneの対処も何かスマートにできるといいかも
        param = []
        for i in range(5):
            param.append(request.form.get(str(i + 1)));
        accuracy = main.ml(requirements_data,target,model,param)
        return render_template('result_page.html',radio=radio_data,target=target,accuracy=accuracy,param=param)
    except Exception as e:
        data = pd.read_csv('data.csv')
        columns_nulldata = main.serch_null(data)[0]#nullのあるカラムと個数を返す
        columns_name = main.serch_null(data)[1]
        return render_template('selection_page.html',columns_name=columns_name,columns_nulldata=columns_nulldata,detail_error=e)
        #return render_template("error_page.html",detail_error=e)




#おまじない
if __name__ == "__main__":
    app.run(debug=True)