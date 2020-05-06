#sudo apt-get install wkhtmltopdf
#pip3 install pdfkit

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
from IPython.utils.capture import capture_output
import base64
import pdfkit
import datetime

import os

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def write_pdf(html_str, file_name):
    options = {'quiet': ''}
    pdfkit.from_string(html_str,'./Printout/'+str(datetime.datetime.now())[:16].replace(' ','_').replace(':','-').replace('.','-')+'_'+file_name, options)

def write_html(html_str, file_name):
    with open(file_name, 'w') as f:
        f.write(html_str)

@magics_class
class export_magic(Magics):
    @line_cell_magic
    def export(self, line=None, cell=None):
        filename = 'temp'
        export_std_out = False
        export_outputs = True
        if line is not None:
            par = line.split(' ')
            if len(par)>0:
                filename = par[0]
            if len(par)>1:
                export_std_out = str2bool(par[1])
            if len(par)>2:
                export_outputs = str2bool(par[2])
        if cell is not None:
            with capture_output(stdout=True, stderr=True, display=True) as io:
                self.shell.run_cell(cell)

            html_str = '<!DOCTYPE html><head><meta name="pdfkit-page-size" content="A6"/> <meta name="pdfkit-orientation" content="Landscape"/><html><body>'
            if export_std_out:
                for ln in io.stdout.splitlines():
                    if ln.find('\x1b\x5b\x32\x4b') == -1:
                        html_str += '<div>%s</div>'%ln
            if export_outputs:
                for o in io.outputs:
                    if 'image/svg+xml' in o.data:
                        img = o.data['image/svg+xml']
                        html_str+='<div>%s</div>'%img
                    elif 'image/png' in o.data:
                        img = o.data['image/png']
                        img_base64 = base64.b64encode(img).decode("utf-8")
                        html_str+='<div><img src="data:image/png;base64,' + img_base64 + '"></div>'
                    
            html_str +='</body></html>'
            if filename.find('.pdf') != -1:
                write_pdf(html_str, filename)
            elif filename.find('.html') != -1:
                write_html(html_str, filename)
            else:
                write_pdf(html_str, filename+'.pdf')
            io.show()

    @line_magic
    def print(self, line):
        filename = 'temp.pdf'
        papersze = 'a6'
        par = line.split(' ')
        if len(par)>0 and par[0]:
            filename = par[0]
        if len(par)>1 and par[1]:
            papersze = par[1]
        cmd = "lpr -o media=%s %s"%(papersze,filename)
        returned_value = os.system(cmd)
        return returned_value==0

get_ipython().register_magics(export_magic)