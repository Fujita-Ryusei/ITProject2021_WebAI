// function displayRadio(key, value) {

//     if (value[1] == 'object') {
//         document.write('<tr>' +
//             '<td>' + '<option value="' + key + '">' + key + ':' + value[0] + '</option>' + '</td>' +
//             '<td>' + '最頻値' + '<input type="radio" id="r1" name=' + key + ' value="mode">' + '</td>' +
//             '<td>' + '中央値' + '<input type="radio" id="r1" name=' + key + ' value="med">' + '</td>' +
//             '<td>' + '削除' + '<input type="radio" id="r1" name=' + key + ' value="drop">' + '</td>' +
//             '</tr>');
//     } else {
//         document.write('<tr>' +
//             '<td>' + '<option value="' + key + '">' + key + ':' + value[0] + '</option>' + '</td>' +
//             '<td>' + '平均値' + '<input type="radio" id="r1" name=' + key + ' value="ave">' + '</td>' +
//             '<td>' + '最頻値' + '<input type="radio" id="r1" name=' + key + ' value="mode">' + '</td>' +
//             '<td>' + '中央値' + '<input type="radio" id="r1" name=' + key + ' value="med">' + '</td>' +
//             '<td>' + '標準偏差' + '<input type="radio" id="r1" name=' + key + ' value="standard">' + '</td>' +
//             '<td>' + '削除' + '<input type="radio" id="r1" name=' + key + ' value="drop">' + '</td>' +
//             '</tr>');
//     }


// }

function displayRadio() {

    if ('{{value[1]}}' == 'object') {
        document.write('<tr>' +
            '<td>' + '<option value="{{ key }}">' + '{{ key }}:{{ value[0] }}' + '</option>' + '</td>' +
            '<td>' + '最頻値' + '<input type="radio" id="r1" name={{ key }} value="mode">' + '</td>' +
            '<td>' + '中央値' + '<input type="radio" id="r1" name={{ key }} value="med">' + '</td>' +
            '<td>' + '削除' + '<input type="radio" id="r1" name={{ key }} value="drop">' + '</td>' +
            '</tr>');
    } else {
        document.write('<tr>' +
            '<td>' + '<option value="{{ key }}">' + '{{ key }}:{{ value[0] }}' + '</option>' + '</td>' +
            '<td>' + '平均値' + '<input type="radio" id="r1" name={{ key }} value="ave">' + '</td>' +
            '<td>' + '最頻値' + '<input type="radio" id="r1" name={{ key }} value="mode">' + '</td>' +
            '<td>' + '中央値' + '<input type="radio" id="r1" name={{ key }} value="med">' + '</td>' +
            '<td>' + '標準偏差' + '<input type="radio" id="r1" name={{ key }} value="standard">' + '</td>' +
            '<td>' + '削除' + '<input type="radio" id="r1" name={{ key }} value="drop">' + '</td>' +
            '</tr>');
    }


}