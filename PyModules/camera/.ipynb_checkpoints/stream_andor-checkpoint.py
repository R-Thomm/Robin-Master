from .streamer import streamer, stream_client, address, get_key_from_password
from .image import streamer_capture

class streamer_andor(streamer):
    def __init__(self, host='10.5.78.148', port=5555, key = get_key_from_password('Mg+=25')):
        addr = address(proto='tcp' , host=host , port=port)
        streamer.__init__(self, stream_client(addr, 500), key=key)
        
        self.flag_compress=True
        self.flag_encrypt=True
        self.flag_base64=True

    def query_msg_enc(self, cmd):
        cmd_enc = self.pack_cmd(cmd, b_compress=self.flag_compress, b_encrypt=self.flag_encrypt, b_base64=self.flag_base64)
        msg_enc = self.query(cmd_enc)
        return msg_enc

    def query_cmd(self, cmd):
        msg_enc = self.query_msg_enc(cmd)
        msg = self.unpack_cmd(msg_enc, b_compress=self.flag_compress, b_encrypt=self.flag_encrypt, b_base64=self.flag_base64)
        ret = (msg=="OK")
        if not ret:
            print('"%s"'%msg)
        return ret
        
    def query_frame(self, cmd):
        msg_enc = self.query_msg_enc(cmd)
        return self.unpack_array(msg_enc, b_compress=self.flag_compress, b_encrypt=self.flag_encrypt, b_base64=self.flag_base64)

    def capture(self, flag_pop=False):
        if flag_pop:
            cmd = 'video'
        else:
            cmd = 'acquire'
        ret = self.query_frame(cmd)
        if ret is not None and len(ret)==2:
            frame, meta = ret
            return frame, meta
        else:
            return None, None
            
    def get(self, num=1):
        frames = []
        metas = []
        while len(frames) < num:
            frm, meta = self.capture()
            if frm is not None:
                frames.append(frm)
                metas.append(meta)
        return frames, metas

    def video(self):
        cmd = 'video'
        return self.query_frame(cmd)
        
    def acquire(self):
        cmd = 'acquire'
        frame, meta = None, None
        while frame is None:
            frame, meta = self.query_frame(cmd)
        return frame, meta

    def clear(self):
        cmd = 'clear-queue'
        return self.query_cmd(cmd)
        
    def select(self, idx):
        cmd = 'select %i'%(idx)
        return self.query_cmd(cmd)

    def end(self):
        cmd = 'end'
        return self.query_cmd(cmd)
        
    def set_cool(self, state):
        if state:
            cmd = 'cool-on'
        else:
            cmd = 'cool-off'
        return self.query_cmd(cmd)

    def set_open_shutter(self):
        cmd = 'open-shutter'
        return self.query_cmd(cmd)

    def set_full_image(self):
        cmd = 'full-image'
        return self.query_cmd(cmd)

    def set_trigger(self, mode):
        cmd = 'trigger %i'%(mode)
        return self.query_cmd(cmd)

    def set_temperature(self, T):
        cmd = 'temperature %i'%(T)
        return self.query_cmd(cmd)

    def set_pre_gain(self, gain):
        cmd = 'pre-gain %i'%(gain)
        return self.query_cmd(cmd)

    def set_emccd_mode(self, mode):
        cmd = 'emccd-mode %i'%(mode)
        return self.query_cmd(cmd)

    def set_emccd_gain(self, gain):
        cmd = 'emccd-gain %i'%(gain)
        return self.query_cmd(cmd)

    def set_exposure(self, t):
        cmd = 'exposure %f'%(t)
        return self.query_cmd(cmd)

    def set_shutter(self, typ, mode):
        cmd = 'shutter %i %i'%(typ, mode)
        return self.query_cmd(cmd)

    def set_orientation(self, iRotate, iHFlip, iVFlip):
        cmd = 'orientation %i %i %i'%(iRotate, iHFlip, iVFlip)
        return self.query_cmd(cmd)

    def set_roi(self, hbin, vbin, hstart, hend, vstart, vend):
        cmd = 'roi %i %i %i %i %i %i'%(hbin, vbin, hstart, hend, vstart, vend)
        return self.query_cmd(cmd)

if __name__ == "__main__":
    import threading
    import sys
    key = get_key_from_password('Mg+=25')
    #addr_host = '10.5.78.171'
    #addr_host = 'localhost'
    #addr_host = '10.5.78.175'
    addr_host = '10.5.78.148'
    if len(sys.argv)>1:
        addr_host = sys.argv[1]
    print('host', addr_host)

    cam = streamer_andor(host=addr_host, port=5555, key=key)    

    cap = streamer_capture(cam, terminal=True, plot_width=None)
    
    low, high = 990,1080
    #low, high = None, None
    x = threading.Thread(target=cap.video, args=(low, high), kwargs={'do_wait':False})
    x.start()

    while True:
        text = input("Command: ")
        if text=='q':
            cap.stop()
            break
        elif text=='p':
            frame, meta = cam.acquire()
            print(meta)
        elif text:
            print(cam.query_cmd(text))

    x.join()

