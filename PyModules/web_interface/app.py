from flask import Flask, render_template, request
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

from flask_socketio import SocketIO, join_room, emit
socketio = SocketIO(app)

import json
import matplotlib.pyplot as plt
import random
from io import StringIO
import numpy as np
from threading import Thread

@app.before_request
def befreq():
    global locked
    if locked:
        return "LOCKED!"

@app.route("/")
def index():
    form_script = '<option value ="none">-</option>'
    for scr in l_script:
        form_script += '<option value ="%s">%s</option>'%(scr,scr)

    form_ip = '<tr><th>Name</th><th>Value</th></tr>'
    for ip,ip_val in zip(l_ip,l_ip_value):
        form_ip += '<tr><td>%s</td><td>%s</td></tr>' % (ip,ip_val)
    form_ip = '<table id="ip_list">%s</table>'%form_ip

    html = render_template("index.html", form_script=form_script, form_ip=form_ip)
    return html

@app.route("/run/", methods=['GET'])
def run_method():
    script = request.args.get('script', None)
    return run_url(script)

@app.route("/run/<script>")
def run_url(script):
    global svg_dta, running
    response = {"status" : False}
    if (script is not None):
        svg_dta = []
        s_id, name = eios.add(script,plot_callback)
        running = bool(s_id>-1)
        response = {"status":running, "script":script}
        
    return json.dumps(response)

@app.route("/stop/<sid>")
def stop_sid(sid=None):
    if sid is not None:
        sid = float(sid)
    stat = eios.stop(sid)
    return json.dumps({"status":stat})

@app.route("/stop/")
def stop():
    return stop_sid(sid=None)

@app.route("/set/<var1>/<var2>")
def set(var1,var2):
    stat = [var1,var2]
    print(var1, var2, file=sys.stderr)

    stat = eios.set(var1,var2)
    global l_script, l_ip, l_ip_value
    l_script, l_ip, l_ip_value = get_eios_lists()

    return json.dumps({"status":stat})

@socketio.on('connect', namespace='/update_graph')
def graph_connect():
    print('graph connect', file=sys.stderr)

@socketio.on('disconnect', namespace='/update_graph')
def graph_disconnect():
    print('graph disconnect', file=sys.stderr)

@app.route("/refresh_graph/")
def refresh_graph():
    global svg_dta, running
    response = {"status":bool(svg_dta), "svg":svg_dta, "run":running}
    return json.dumps(response)

def plot_callback(s_id, data_ctr, last_call=False):
    global svg_dta, running
    running = not last_call
    #print('plot_callback', bool(data_ctr), (s_id>-1), file=sys.stderr)
    if bool(data_ctr) and (s_id>-1):
        fig = eios.plot_data(data_ctr)
        svg_dta = plot2svg(fig)
        plt.close(fig=fig)
        response = {"status":True, "svg":svg_dta, "run":running}
        socketio.emit('update_graph', json.dumps(response), namespace='/update_graph')

    #print('update id=%i, valid=%i, last_call=%i'%(s_id, (bool(data_ctr) and s_id>-1), last_call), file=sys.stderr)
    return True

def plot2svg(fig):
    #https://stackoverflow.com/questions/5453375/matplotlib-svg-as-string-and-not-a-file
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data

    svg_data = imgdata.getvalue()  # this is svg data
    imgdata.close()
    return svg_data

import sys
sys.path.insert(0, './../../')
from PyModules.eios import EIOS_META_SCRIPT

if __name__ == '__main__':
    global eios
    eios = EIOS_META_SCRIPT()

    def get_eios_lists():
        l_script = eios.list_script()
        l_ip = eios.list_ip()
        l_ip_value = []
        for ip in l_ip:
            l_ip_value.append(eios.get(ip))
        return l_script, l_ip, l_ip_value

    global l_script, l_ip, l_ip_value
    l_script, l_ip, l_ip_value = get_eios_lists()

    global locked, svg_dta, running
    locked = False
    svg_dta = []
    running = False

    # Debug
    host = '10.5.78.175'
    #host = '127.0.0.1'
    port = 8080
    socketio.run(app, host=host, port=port, use_reloader=False, debug=True)
    #app.run(host=host, port=port, use_reloader=False, debug=True)

    '''
    # Production
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    '''

'''
    set(name,value)
    get(name)
    set_profile(profile_name,shim_name,value)
    get_profile(profile_name,shim_name)
    ttl_ME(channel,state)
    ttl_PB(channel,state)
    trigger(channel,t_width)
    pdq_trigger(trigger_cnt = 1) # 1 = one trigger, 2 = two trigger
    set_queue( run=True, interleaved=False, random=False)
    run(script_name, p_start=None, p_stop=None, p_numofpoints=None, p_expperppoint=None, rnd_sampled=None, live_plots_flag=None)
    add(script_name, func, p_start=None, p_stop=None, p_numofpoints=None, p_expperppoint=None, rnd_sampled=None, live_plots_flag=None)
    get_parameter(script_name)
    list_script()
    list_ip()
    stop(id=None)
    end()
    exit()
    plot_data(name,data_ctr)
'''

