<?php

// 出力情報の設定
header("Content-Type: application/octet-stream");
header("Content-Disposition: attachment; filename=GRAYCODE.csv");
header("Content-Transfer-Encoding: binary");

// 変数の初期化
$member = array();
$csv = null;

// 出力したいデータのサンプル
$member = array(
	array(
		'id' => 1,
		'name' => '山田太郎',
		'furigana' => 'やまだたろう',
		'email' => 'taroyamada@sample.com'
	),
	array(
		'id' => 3,
		'name' => '加藤明美',
		'furigana' => 'かとうあけみ',
		'email' => 'akemikato@sample.com'
	),
	array(
		'id' => 5,
		'name' => '佐藤健夫',
		'furigana' => 'さとうたけお',
		'email' => 'takeosato@sample.com'
	)
);

// 1行目のラベルを作成
$csv = '"ID","氏名","ふりがな","メールアドレス"' . "\n";

// 出力データ生成
foreach( $member as $value ) {
	$csv .= '"' . $value['id'] . '","' . $value['name'] . '", "' . $value['furigana'] . '","' . $value['email'] . '"' . "\n";
}

// CSVファイル出力
echo $csv;
return;