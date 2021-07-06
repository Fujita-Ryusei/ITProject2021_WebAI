

const parameters_area = document.getElementsByClassName("parameters_area")[0];
//const parameters_area = document.getElementById('parameters_area');

// name をモデルで共通にする！
// 欠点はモデルのパラメータ名を受け取れなくなるので順番ごとにパラメータを代入する
// name は1.2.3とかでやればモデルごとにパラメータの表示数を変えられる
// post先でモデルの種類に応じてfor文でrequestをとれる

//-------------------------------------------------------
//https://ichi.pro/randamufuxoresutohaipa-parame-tachu-ningu-no-bigina-zugaido-77596161963319
//https://qiita.com/FujiedaTaro/items/61ded4ea5643a6204317
//https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
//RandomForestClassifier
//n_estimators
var td1_n_estimators = document.createElement("td");
var n_estimators_label = document.createTextNode("n_estimators");
var n_estimators = document.createElement("input");
n_estimators.setAttribute("name", "1");
n_estimators.setAttribute("type", "number");
n_estimators.setAttribute("value", "100");
n_estimators.setAttribute("max", "1000");
n_estimators.setAttribute("min", "10");
td1_n_estimators.appendChild(n_estimators_label);
td1_n_estimators.appendChild(n_estimators);

//criterion
var td2_criterion_classifier = document.createElement("td");
var criterion_label = document.createTextNode("criterion");
var criterion = document.createElement('select');
var gini = document.createElement("option");
var entropy = document.createElement("option");
gini.setAttribute("value", "gini");
gini.appendChild(document.createTextNode("gini"));
entropy.setAttribute("value", "entropy");
entropy.appendChild(document.createTextNode("entropy"));
criterion.setAttribute("name", "2");
criterion.appendChild(gini);
criterion.appendChild(entropy);
td2_criterion_classifier.appendChild(criterion_label);
td2_criterion_classifier.appendChild(criterion);

//max_depth
var td3_max_depth = document.createElement("td");
var max_depth_label = document.createTextNode("max_depth");
var max_depth = document.createElement("input");
max_depth.setAttribute("type", "checkbox");
//max_depth.setAttribute("name","3");
max_depth.setAttribute("value", "");
max_depth.setAttribute("onchange", "maxDepthFunc();"); //maxDepthFunc()で選択したときにmax_depth_intを表示
var max_depth_int = document.createElement("input");
max_depth_int.setAttribute("type", "number");
max_depth_int.setAttribute("name", "3");
max_depth_int.setAttribute("value", "100");
max_depth_int.setAttribute("max", "1000");
max_depth_int.setAttribute("min", "1");
td3_max_depth.appendChild(max_depth_label);
td3_max_depth.appendChild(max_depth);

//min_samples_split
var td4_min_samples_split = document.createElement("td");
var min_samples_split_label = document.createTextNode("min_samples_split");
var min_samples_split = document.createElement("input");
min_samples_split.setAttribute("type", "number");
min_samples_split.setAttribute("name", "4");
min_samples_split.setAttribute("value", "10");
min_samples_split.setAttribute("max", "2000");
min_samples_split.setAttribute("min", "10");
td4_min_samples_split.appendChild(min_samples_split_label);
td4_min_samples_split.appendChild(min_samples_split);

//max_leaf_nodes
var td5_max_leaf_nodes = document.createElement("td");
var max_leaf_nodes_label = document.createTextNode("max_leaf_nodes");
var max_leaf_nodes = document.createElement("input");
max_leaf_nodes.setAttribute("type", "checkbox");
//max_leaf_nodes.setAttribute("name","5");
max_leaf_nodes.setAttribute("value", "");
max_leaf_nodes.setAttribute("onchange", "maxLeafNodesFunc();");
var max_leaf_nodes_int = document.createElement("input");
max_leaf_nodes_int.setAttribute("type", "number");
max_leaf_nodes_int.setAttribute("name", "5");
max_leaf_nodes_int.setAttribute("value", "15");
max_leaf_nodes_int.setAttribute("max", "200");
max_leaf_nodes_int.setAttribute("min", "0");
td5_max_leaf_nodes.appendChild(max_leaf_nodes_label);
td5_max_leaf_nodes.appendChild(max_leaf_nodes);


//-----------------------------------------------------------------
//RandomForestRegressor
//n_estimators td1_n_estimators

//criterion
var td2_criterion_regressor = document.createElement("td");
var criterion_label = document.createTextNode("criterion");
var criterion = document.createElement('select');
var mse = document.createElement("option");
var mae = document.createElement("option");
mse.setAttribute("value", "mse");
mse.appendChild(document.createTextNode("mse"));
mae.setAttribute("value", "mae");
mae.appendChild(document.createTextNode("mae"));
criterion.setAttribute("name", "2");
criterion.appendChild(mse);
criterion.appendChild(mae);
td2_criterion_regressor.appendChild(criterion_label);
td2_criterion_regressor.appendChild(criterion);

//max_depth td3_max_depth

//min_samples_split td4_min_samples_split

//max_leaf_nodes td5_max_leaf_nodes


//-----------------------------------------------------------------
//XGBoost
//https://xgboost.readthedocs.io/en/latest/parameter.html
//https://qiita.com/FJyusk56/items/0649f4362587261bd57a

//max_depth
var td1_max_depth_xgb = document.createElement("td");
var max_depth_xgb_label = document.createTextNode("max_depth");
var max_depth_xgb = document.createElement("input");
max_depth_xgb.setAttribute("name", "1");
max_depth_xgb.setAttribute("type", "number");
max_depth_xgb.setAttribute("value", "6");
max_depth_xgb.setAttribute("max", "1000");
max_depth_xgb.setAttribute("min", "0");
td1_max_depth_xgb.appendChild(max_depth_xgb_label);
td1_max_depth_xgb.appendChild(max_depth_xgb);

//eta
var td2_eta_xgb = document.createElement("td");
var eta_xgb_label = document.createTextNode("eta");
var eta_xgb = document.createElement("input");
eta_xgb.setAttribute("step", "0.1");
eta_xgb.setAttribute("name", "2");
eta_xgb.setAttribute("type", "number");
eta_xgb.setAttribute("value", "0.3");
eta_xgb.setAttribute("max", "1");
eta_xgb.setAttribute("min", "0");
td2_eta_xgb.appendChild(eta_xgb_label);
td2_eta_xgb.appendChild(eta_xgb);

//objective
var td3_objective_xgb = document.createElement("td");
var objective_xgb_label = document.createTextNode("objective");
var objective_xgb = document.createElement('select');

var linear = document.createElement("option");
linear.setAttribute("value", "reg:linear");
linear.appendChild(document.createTextNode("linear"));
// logistic はXGB側で処理の分岐など
// とりあえずプロトタイプなので保留(気が向いたらやるかも？)
//var logistic = document.createElement("option");
//logistic.setAttribute("value", "reg:logistic");
//logistic.appendChild(document.createTextNode("logistic"));
//var b_logistic = document.createElement("option");
//b_logistic.setAttribute("value", "binary:logistic");
//b_logistic.appendChild(document.createTextNode("b_logistic"));
var softmax = document.createElement("option");
softmax.setAttribute("value", "multi:softmax");
softmax.appendChild(document.createTextNode("softmax"));

objective_xgb.setAttribute("name", "3");
objective_xgb.appendChild(linear);
//objective_xgb.appendChild(logistic);
//objective_xgb.appendChild(b_logistic);
objective_xgb.appendChild(softmax);
td3_objective_xgb.appendChild(objective_xgb_label);
td3_objective_xgb.appendChild(objective_xgb);

//num_round
var td4_num_round_xgb = document.createElement("td");
var num_round_xgb_label = document.createTextNode("num_round");
var num_round_xgb = document.createElement("input");
num_round_xgb.setAttribute("name", "4");
num_round_xgb.setAttribute("type", "number");
num_round_xgb.setAttribute("value", "10");
num_round_xgb.setAttribute("max", "100");
num_round_xgb.setAttribute("min", "1");
td4_num_round_xgb.appendChild(num_round_xgb_label);
td4_num_round_xgb.appendChild(num_round_xgb);

//num_class


function radioSelect(model) {

    for (let i = parameters_area.childNodes.length - 1; i >= 0; i--) {
        parameters_area.removeChild(parameters_area.childNodes[i]);
    }
    if (model.value == "RandomForestClassifier") {
        parameters_area.appendChild(td1_n_estimators);
        parameters_area.appendChild(td2_criterion_classifier);
        parameters_area.appendChild(td3_max_depth);
        parameters_area.appendChild(td4_min_samples_split);
        parameters_area.appendChild(td5_max_leaf_nodes);
    } else if (model.value == "RandomForestRegressor") {
        parameters_area.appendChild(td1_n_estimators);
        parameters_area.appendChild(td2_criterion_regressor);
        parameters_area.appendChild(td3_max_depth);
        parameters_area.appendChild(td4_min_samples_split);
        parameters_area.appendChild(td5_max_leaf_nodes);
    } else if (model.value == "XGBoost") {
        parameters_area.appendChild(td1_max_depth_xgb);
        parameters_area.appendChild(td2_eta_xgb);
        parameters_area.appendChild(td3_objective_xgb);
        parameters_area.appendChild(td4_num_round_xgb);
    }

}


//とりあえず試作ではbool
var max_depth_flag = new Boolean(true);
function maxDepthFunc() {
    if (max_depth_flag) {
        td3_max_depth.appendChild(max_depth_int);
        max_depth_flag = false;
    } else {
        td3_max_depth.removeChild(td3_max_depth.childNodes[2]);
        max_depth_flag = true;
    }
}
var max_leaf_nodes_flag = new Boolean(true);
function maxLeafNodesFunc() {
    if (max_leaf_nodes_flag) {
        td5_max_leaf_nodes.appendChild(max_leaf_nodes_int);
        max_leaf_nodes_flag = false;
    } else {
        td5_max_leaf_nodes.removeChild(td5_max_leaf_nodes.childNodes[2]);
        max_leaf_nodes_flag = true;
    }
}

