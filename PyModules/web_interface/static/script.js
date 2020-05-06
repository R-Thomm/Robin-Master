function httpGetAsync(theUrl, callback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", theUrl, true); // true for asynchronous 
    xmlHttp.send(null);
}

function get_url_status(url,status_field){
	httpGetAsync(url, function(response){
	    parsedresponse = JSON.parse(response);
        document.getElementById(status_field).innerHTML = parsedresponse.status;
    });
}

function run_script() {
    var script = document.getElementById("select_script").value;
    get_url_status("/run/"+script,"result_script");
    timer_graph();
}

function stop_script() {
    get_url_status("/stop","result_script");
}

function parse_graph(response){
    parsedresponse = JSON.parse(response)
    if (parsedresponse.status){
	    document.getElementById("graph_script").innerHTML = parsedresponse.svg;
    }
    return parsedresponse.run;
}

function timer_graph(){
	//console.log(1);
    httpGetAsync("/refresh_graph/", function(response){
        parsedresponse = JSON.parse(response)
        if (parsedresponse.status){
	        document.getElementById("graph_script").innerHTML = parsedresponse.svg;
            if (parsedresponse.run) {
                setTimeout(timer_graph,500);
            }
        }
    });
}

function connect_graph(){
    // start up the SocketIO connection to the server - the namespace 'update_graph'
    var url = 'http://' + document.domain + ':' + location.port + '/update_graph';
    var socket = io.connect(url);
    // callback triggered i server emits 'update_graph' event
    socket.on('update_graph', parse_graph);
}

function table_set(e){
    var ip_name = $("#ip_list tr.selected td:first").html();
    var ip_value = document.getElementById("ip_value").value;
    if (typeof ip_name !== 'undefined') {
        get_url_status("/set/" + ip_name + "/" + ip_value, "result_ip");
        alert(ip_name+" = "+ip_value)
        document.location.reload();
    } else {
        alert("Please select IP!")
    }
}

function table_select(){
    $(this).addClass('selected').siblings().removeClass('selected');    
    /*
    var ip_name=$(this).find('td:first').html();
    var ip_value=$(this).find('td:last').html();
    alert(ip_name+" = "+ip_value)
    */
}

function table_connect(){
    $("#ip_list tr").click(table_select);
    $('.ok').on('click', table_set);
}

function doc_ready(){
    connect_graph();
    table_connect();
}

// Client Side Javascript to receive numbers.
$(document).ready(doc_ready);

