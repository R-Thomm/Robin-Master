import time
import datetime
import numpy as np

from threading import Lock

from copy import copy
import zlib
import msgpack
import msgpack_numpy as m
import scipy.constants as constants

import cv2
from PIL import ImageFont, ImageDraw, Image

from matplotlib import cm
import matplotlib.pyplot as plt

font_mono = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

def image_text(A, txt, pos=(0, 0), font_file=font_mono, font_size=50, **kwargs):
    try:
        font = ImageFont.truetype(font_file, font_size)
    except:
        font = ImageFont.truetype("arial.ttf", font_size)

    pil_im = Image.fromarray(A, **kwargs)
    draw = ImageDraw.Draw(pil_im)
    draw.text(pos, txt, font=font)
    A = np.array(pil_im)
    #cv2.putText(A, fps, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return A

def mark(A, xy, txt=None, pos=(0, 0), font_file=font_mono, font_size=50, **kwargs):
    pil_im = Image.fromarray(A)
    draw = ImageDraw.Draw(pil_im)
    draw.rectangle(xy, **kwargs)

    if txt is not None:
        try:
            font = ImageFont.truetype(font_file, font_size)
        except:
            font = ImageFont.truetype("arial.ttf", font_size)
        draw.text(pos, txt, font=font)

    A = np.array(pil_im)
    return A

def image_sc(A, _min, _max, cmap):
    A = (A.astype('float')-_min)/(_max-_min)
    A = cmap(A, bytes=True)
    return A

def roi(frame, x, y, width, height):
    return frame[x:x+width, y:y+height]

def frame_limits(frame):
    return float(np.amin(frame,axis=None)), float(np.amax(frame,axis=None))

# cmap: viridis, cividis, inferno
def scale(frame, low, high, cmap=cm.inferno, flag_BGRA=True):
    frame = np.squeeze(frame)
    if len(frame.shape)<3:
        frame = image_sc(frame, low, high, cmap=cmap)
        if flag_BGRA:
            cv2.cvtColor(frame,cv2.COLOR_RGBA2BGRA,frame)
    return frame

def add_cmap(frame, low, high, length_ratio=[.05,.8], pos_ratio=[0.,.2]):
    size = [min(int(a*b),a) for a,b in zip(frame.shape,length_ratio)]
    pos_start = [min(int(a*p),a) for a,p in zip(frame.shape,pos_ratio)]
    if size[0]>size[1]:
        smap = np.reshape(np.linspace(low,high,size[0]),(size[0],1))
        smat = np.repeat(smap, size[1], axis=1)
    else:
        smap = np.reshape(np.linspace(low,high,size[1]),(1,size[1]))
        smat = np.repeat(smap, size[0], axis=0)
    tframe = frame.copy()
    pos_end = [min(int(a+b),s) for s,a,b in zip(frame.shape,size,pos_start)]
    tframe[pos_start[0]:pos_end[0],pos_start[1]:pos_end[1]] = smat
    return tframe, pos_start, pos_end

def dict_decode(_dict):
    _new_dict = copy(_dict)
    for i,key in enumerate(_dict):
        if isinstance(_dict, dict):
            new_key = key
            if isinstance(key, bytes):
                new_key = key.decode('utf-8')
                _new_dict[new_key] = _new_dict.pop(key)
            if isinstance(_dict[key], dict) or isinstance(_dict[key], list):
                _new_dict[new_key] = dict_decode(_dict[key])
        elif isinstance(_dict, list):
            if isinstance(key, dict) or isinstance(key, list):
                _new_dict[i] = dict_decode(key)
    return _new_dict

import seaborn as sns

def show_image(frame, meta, size=3.):
    sc = frame.shape[1]/frame.shape[0]
    fig, ax = plt.subplots(figsize=(size*sc, size))        
    sns.heatmap(frame, cmap="inferno")

    timestr = str(datetime.datetime.fromtimestamp(meta['timestamp']))
    txt = timestr
    txt += '\nExp. (ms): '+str(round(meta['exposure'],3))
    txt += ' - Gain: '+str(round(meta['emccd_gain'],1))
    txt += ' - T (degC): '+str(round(meta['temperature'],1))
    plt.title(txt)
        
    plt.xlabel('z - direction (px)')
    plt.ylabel('x - direction (px)')
    plt.grid(True, color='w', alpha = .25)
    return fig, ax
    
    '''
    #plt.show()
    
    #plt.gcf().canvas.flush_events()
    #plt.show(block=False)
    '''

import skimage.feature as sk

def detect_blobs(frame, threshold=.005):
    blobs = sk.blob_dog(frame, max_sigma=5, threshold=threshold)
    blobs = np.array(blobs)
    return blobs

def analyse_blobs(blobs, px_to_um):
    center_of_gravity = (0,0)
    angle = 0
    ion_dist = [0,0]
    if len(blobs)>1:
        x_cog = np.sum(blobs[0:len(blobs), 0])/len(blobs)
        y_cog = np.sum(blobs[0:len(blobs), 1])/len(blobs)
        center_of_gravity = (x_cog, y_cog)
        angle = np.arctan(-(blobs[0, 0]-blobs[-1, 0])/(blobs[0, 1]-blobs[-1, 1]))/(2+np.pi)*360
        ion_dist_px=((blobs[0,0]-blobs[-1,0])**2+(blobs[0,1]-blobs[-1,1])**2)**0.5
        ion_dist = (ion_dist_px*px_to_um/(len(blobs)-1), px_to_um*blobs[1,2]/5)
    return center_of_gravity, angle, ion_dist

def get_modfreq_dist(N, M, dist):
        '''Calculate Lowest Mode Frequency from Measured distance
        N = No of Ions
        M = Mass number of an Ion
        dist = Distance between Ions in µm
        '''
        if N==1:
            ''
            
        if N>1:
            
            freq = np.sqrt(((constants.elementary_charge)**2 /(4*np.pi*constants.epsilon_0*M*constants.u)*(2.018/(N**0.559))**3)/(dist[0]*1E-6)**3)/(2*np.pi)
            freq_err = freq - np.sqrt(((constants.elementary_charge)**2 /(4*np.pi*constants.epsilon_0*M*constants.u)*(2.018/(N**0.559))**3)/((dist[0]+dist[1])*1e-6)**3)/(2*np.pi)
            lf = (freq,freq_err)
            return lf
            

def show_ion_text(ax, blobs, center_of_gravity, angle, ion_dist, modefreq, rel_pos=(1.3, 1.)):
    ion_pos_txt = 'Ions:\n-----\n'
    for blob in blobs:
        y, x, r = blob
        ion_pos_txt += 'Pos. (x, z): (%i, %i)\n'%(y, x)

    if len(blobs)>1:
        ion_pos_txt += '\nC-o-G (x, z): (%.1f, %.1f)\n'%center_of_gravity
        ion_pos_txt += 'In-plane angle (deg): %.2f'%(angle)
        ion_pos_txt += '\n\nAvg. inter-ion distance (µm): %.1f +/- %.1f'%ion_dist
        ion_pos_txt += '\n\nAvg. Lowest Mode frequency (Hz): %.1f +/- %.1f'%modefreq
    x_rel, y_rel = rel_pos
    ax.text(x_rel, y_rel, ion_pos_txt, 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top')
    return ion_pos_txt

def show_blobs( ax, blobs, center_of_gravity, px_to_um, scale_width_pixel, 
                offset_scale=(5,5), font_size=12, scale_r=2):
    for blob in blobs:
        y, x, r = blob
        r *= scale_r
        c = plt.Circle((x, y), r, color='red', linewidth=1.5, fill=False)
        ax.add_patch(c)

    # center of gravity
    if len(blobs)>1:
        ax.axhline(center_of_gravity[0])
        ax.axvline(center_of_gravity[1])

    # scale
    length_scale = scale_width_pixel*px_to_um
    length_scale_text = '%.1f µm'%length_scale

    # scale line
    x0_scale, y0_scale = offset_scale
    ax.hlines(  y=y0_scale, xmin=x0_scale, 
                xmax=x0_scale+scale_width_pixel, 
                linewidth=2.5, color='w')
    ax.text(x0_scale, y0_scale+font_size, 
            length_scale_text, fontsize=font_size, 
            verticalalignment='top', color='w')

class visualizer_opencv(object):
    def __init__(self, plot_width=None, title='stream'):
        self.title = title
        self.plot_width = plot_width
        self.plot_height = None

    def __initialize(self, frame):
        height, width, depth = frame.shape[0:3]
        if self.plot_width is None:
            self.plot_width = width
            self.plot_height = height
            return
        self.plot_width = int(self.plot_width)
        self.plot_height = int(self.plot_width*height/width)
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.plot_width, self.plot_height)

    def show(self, frame):
        if self.plot_height is None:
            self.__initialize(frame)
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)

    def scale(self, frame, low, high):
        return scale(frame, low, high, flag_BGRA=True)

    def reset(self):
        #close the image window
        cv2.destroyWindow(self.title)
        #cv2.DestroyAllWindows()

from bokeh.io import push_notebook, output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource

class visualizer_bokeh(object):
    def __init__(self, plot_width=None):
        self.plot_width = plot_width
        self.plot_height = None
        output_notebook(verbose=False, hide_banner=True);
        self.reset()

    def reset(self):
        self.source = None

    def __initialize(self, frame):
        height, width, depth = frame.shape[0:3]
        self.source = ColumnDataSource(data={'image':[np.flipud(frame)]})
        if self.plot_width is None:
            self.plot_width = width
        self.plot_width = int(self.plot_width)
        self.plot_height = int(self.plot_width*height/width)
        bokeh_plot = figure(x_range=(0,width), y_range=(height, 0), x_axis_location='above', output_backend="webgl", width=self.plot_width, height=self.plot_height, match_aspect=True)
        bokeh_plot.image_rgba('image', source=self.source, x=0, y=height, dw=width, dh=height)

        #from bokeh.models.callbacks import CustomJS
        #callback = CustomJS(code="""console.log('Tap event occurred at ' + cb_obj.x + '|' + cb_obj.y)""")
        #bokeh_plot.js_on_event('tap', callback)
        self.bokeh_handle = show(column(bokeh_plot), notebook_handle=True)

    def __update(self, frame):
        self.source.data['image'] = [np.flipud(frame)]
        push_notebook(handle=self.bokeh_handle)

    def scale(self, frame, low, high):
        return scale(frame, low, high, flag_BGRA=False)

    def show(self, frame):
        if self.source is None:
            self.__initialize(frame)
        self.__update(frame)

class capture_msgpack():
    def __init__(self):
        self.data_dict = {
            'created': 0,
            'metas': [],
            'frames': [],
        }
        self.filename = ''

    def open(self, path='./data/', filename=''):
        if filename:
            self.data_dict = self.load(filename)
        else:
            ts = datetime.datetime.now()
            filename = path+'capture_' + ts.strftime('%Y-%m-%d_%H-%M-%S') + '.msgpack'
            self.data_dict['created'] = ts.timestamp()
        self.filename = filename
        return filename

    def append_fast(self, frame, meta={}):
        self.data_dict['metas'].append(meta)
        self.data_dict['frames'].append(frame)

    def append(self, frame, meta={}):
        self.append_fast(frame, meta)
        self.save(self.filename, self.data_dict)

    def get(self):
        frames = self.data_dict['frames']
        metas = self.data_dict['metas']
        ts = self.data_dict['created']
        return frames, metas, ts

    def save(self, filename, data_dict, level=-1):
        packed_byte = msgpack.packb(data_dict, default=m.encode)
        comprs_byte = zlib.compress(packed_byte, level=level)
        with open(filename, 'wb') as outfile:
            outfile.write(comprs_byte)

    def load(self, filename):
        with open(filename, 'rb') as data_file:
            comprs_byte = data_file.read()
        packed_byte = zlib.decompress(comprs_byte)
        data_dict = msgpack.unpackb(packed_byte, object_hook=m.decode)
        data_dict = dict_decode(data_dict)
        return data_dict
        
    def save_image(self, frame, meta, path='./', filename=''):
        if len(frame.shape)<3 or frame.shape[2] == 1:
            im_type = 'L'
        elif frame.shape[2] == 3:
            im_type = 'RGB'
        else:
            raise Exception('Unknown shape')
    
        if not filename:
            ts = meta['timestamp']
            tme = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(ts))
            filename = 'capture_' + tme + '.png'

        fullfilename = path+filename
        im = Image.fromarray(frame)
        im.save(fullfilename)
        return fullfilename

    def close(self):
        if self.filename:
            self.save(self.filename, self.data_dict)

    def __del__(self):
        self.close()
      
class marker(object):
    def __init__(self, callback=None):
        self.mutex_marker = Lock()
        self.__reset()

    def __reset(self):
        with self.mutex_marker:
            self.marker_boxes = {}
            self.marker_index = 0

    def __remove(self, idx):
        with self.mutex_marker:
            if idx in self.marker_boxes:
                del self.marker_boxes[idx]

    def __del__(self):
        self.__reset()

    def __default_callback(self, roi):
        return np.max(roi)

    def mark(self, frame, x, y, width, height, txt=None):
        xy = [x, y, x+width, y+height]
        pos = (x,y-20)
        return mark(frame, xy, txt=txt, pos=pos, font_size=15, fill=None, outline=None)

    def marker_set(self, idx, x, y, width, height, callback=None):
        if callback is None or not callable(callback):
            callback = self.__default_callback
        self.marker_boxes[idx] = [[x, y, width, height], callback]

    def marker_add(self, x, y, width, height, callback=None):
        with self.mutex_marker:
            idx = self.marker_index
            self.marker_set(idx, x, y, width, height, callback)
            self.marker_index += 1
            return idx
    
    def marker_remove(self, idx=None):
        if idx is None:
            self.__reset()
        else:
            self.__remove(idx)

    def marker_draw(self, frame_data, frame_canvas):
        with self.mutex_marker:
            for key, entry in self.marker_boxes.items():
                box, callback = entry
                results = callback(roi(frame_data, *box))
                txt='%i'%(key)
                for r in results:
                    txt+= ' %.1f'%(r)
                frame_canvas = self.mark(frame_canvas, *box, txt=txt)
        return frame_canvas

class streamer_capture(capture_msgpack, marker):
    def __init__(self, camera=None, terminal=True, plot_width=None, flag_spectator=True):
        capture_msgpack.__init__(self)
        marker.__init__(self)
        self.loop = False
        self.camera = camera
        if terminal:
            self.plotter = visualizer_opencv(plot_width=plot_width)
        else:
            self.plotter = visualizer_bokeh(plot_width=plot_width)
        self.flag_pop = not flag_spectator
        self.l_ts = 0
        self.str_cooling = '.'

    def show(self, frame):
        self.plotter.show(frame)

    def annotate(self, frame, text, pos=(0,0), font_size=50):
        if len(frame.shape)<3:
            frame = image_text(frame, text, pos=pos, font_size=font_size, mode='RGBA')
        else:
            frame = image_text(frame, text, pos=pos, font_size=font_size)
        return frame

    def annotate_relative(self, frame, text, pos=(0,0), font_size=.1):
        abs_pos = [min(int(a*p),a) for a,p in zip(frame.shape,pos)]
        font_size_abs = int(font_size*frame.shape[1])
        return self.annotate(frame, text, pos=abs_pos, font_size=font_size_abs)

    def image_scale(self, frame, low=None, high=None, sat=None, flag_cmap=True):
        if (high is None and sat is not None):
            high = sat
        if (low is None) or (high is None):
            low, high = frame_limits(frame)

        # add colormap
        if flag_cmap:
            # length_ratio=[.05,.8], pos_ratio=[0.,.2]
            frame, pos, size = add_cmap(frame, low, high, length_ratio=[.9,.05], pos_ratio=[.05,0.])

        # scale: grayscale -> RGBA/BGRA
        frame = self.plotter.scale(frame, low, high)

        # add cmap label
        if flag_cmap:
            # (.07,.01) / (.86,.01)   .03
            frame = self.annotate_relative(frame, '%.0f'%low, pos=(.005,.01), font_size=.02)
            frame = self.annotate_relative(frame, '%.0f'%high, pos=(.005,.96), font_size=.02)

        return frame, low, high

    def capture(self, low, high, font_size, offset):
        frame_raw, meta = self.camera.capture(self.flag_pop)
        
        if frame_raw is None:
            return None, None, 0

        self.c_ts = meta['timestamp']
        dt_capture = (self.c_ts-self.l_ts)
        if dt_capture == 0:
            return None, None, 0
        self.l_ts = self.c_ts

        frame,_,_ = self.image_scale(frame_raw, low, high, meta['saturation'])

        frame = self.marker_draw(frame_raw, frame)

        fps = '%6.2f fps'%(1./dt_capture)
        frame = self.annotate(frame, fps, pos=offset, font_size=font_size)

        if 'temperature' in meta:
            temp = '%6.2f  °C'%(meta['temperature'])
            if 'cooler' in meta and meta['cooler']:
                temp = temp+' *'
            if 'temp_stabilized' in meta:
                temp_stat = meta['temp_stabilized']
                if not (temp_stat==20035 or temp_stat==20036):
                    temp = temp + self.str_cooling
                    self.str_cooling += '.'
                    if len(self.str_cooling)>3:
                        self.str_cooling = '.'
            frame = self.annotate(frame, temp, pos=(offset[0],offset[1]+font_size+5), font_size=font_size)

        if 'exposure' in meta:
            exp = '%6.2f   s'%(meta['exposure'])
            frame = self.annotate(frame, exp, pos=(offset[0],offset[1]+2*font_size+5), font_size=font_size)

        if 'emccd_gain' in meta:
            gain = '%5.0f gain'%(meta['emccd_gain'])
            frame = self.annotate(frame, gain, pos=(offset[0],offset[1]+3*font_size+5), font_size=font_size)

        self.show(frame)

        return frame_raw, meta, dt_capture

    def video(self, low=0, high=None, do_wait=True, callback=None, font_size=15, offset=(30,30)):
        if self.camera is None:
            return

        self.loop = True
        self.l_ts = 0
        while self.loop:
            try:
                t0 = time.time()
                frame_raw, meta, dt_capture = self.capture(low, high, font_size, offset)

                if frame_raw is None:
                    time.sleep(.01)
                    continue
                
                if callable(callback):
                    if not callback(frame_raw, meta):
                        callback=None
                
                if do_wait and self.l_ts>0:
                    # max. possible stream fps
                    dt = time.time()-t0
                    t_wait = max(dt_capture-dt,0)
                    time.sleep(t_wait)
                
            except KeyboardInterrupt:
                self.stop()
                
    def stop(self):
        self.plotter.reset()
        self.loop = False
        
