import time
import zmq
import random
import zlib
import json
import numpy as np
import base64
from threading import Lock

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

def get_key_from_password(password):
	digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
	digest.update(bytes(password, 'utf-8'))
	return base64.urlsafe_b64encode(digest.finalize())

def address(proto , host , port):
    return "%s://%s:%i"%(proto,host,port)

# level: 0..9 or -1
#    level of compression; 
#        0 (Z_NO_COMPRESSION)       no compression
#        1 (Z_BEST_SPEED)           fastest, least compression
#        9 (Z_BEST_COMPRESSION)     slowest, most compression
#       -1 (Z_DEFAULT_COMPRESSION)  default: compromise speed/compression (==6).
def compress(_bytes, level=-1):
    _packed = zlib.compress(_bytes, level=level)
    return _packed

def decompress(_packed):
    _bytes = zlib.decompress(_packed)
    return _bytes

def bytes_2_array(_bytes, _dtype, _shape):
    _array = np.frombuffer(_bytes, dtype=_dtype).reshape(_shape)
    return _array

def array_2_bytes(_array):
    _bytes = _array.tobytes()
    _dtype = str(_array.dtype)
    _shape = _array.shape
    return _bytes, _dtype, _shape

class stream_server(object):
    def __init__(self, addr, recv_timeout=0, send_timeout=0):
        self.addr = addr
        self.recv_timeout = recv_timeout
        self.send_timeout = send_timeout
        self.ctx = zmq.Context()
        self.connect()

    def connect(self):
        self.socket = self.ctx.socket(zmq.REP)
        if self.recv_timeout>0:
            self.socket.setsockopt(zmq.RCVTIMEO,self.recv_timeout)
        if self.send_timeout>0:
            self.socket.setsockopt(zmq.SNDTIMEO,self.send_timeout)
        self.socket.bind(self.addr)

    def reconnect(self):
        self.socket.close()
        self.connect()

    def __del__(self):
        self.socket.close()

class stream_client(object):
    def __init__(self, addr, recv_timeout=0, send_timeout=0):
        self.addr = addr
        self.recv_timeout = recv_timeout
        self.send_timeout = send_timeout
        self.ctx = zmq.Context()
        self.connect()

    def connect(self):
        self.socket = self.ctx.socket(zmq.REQ)
        if self.recv_timeout>0:
            self.socket.setsockopt(zmq.RCVTIMEO,self.recv_timeout)
        if self.send_timeout>0:
            self.socket.setsockopt(zmq.SNDTIMEO,self.send_timeout)
        self.socket.connect(self.addr)

    def reconnect(self):
        self.socket.close()
        self.connect()

    def __del__(self):
        self.socket.close()

class streamer():
    def __init__(self, stream, key='', level=-1):
        self.stream = stream
        self.count_failed = 0

        # base64url, AES 128
        if not key:
            key = Fernet.generate_key()
        self.key = key
        #print('Using key: %s'%self.key)
        self.crypter = Fernet(self.key) # Fernet: symmetric authenticated cryptography

        self.level = level
        self.mutex = Lock()

    def encrypt(self, _bytes):
        _token = self.crypter.encrypt(_bytes)
        return _token

    def decrypt(self, _token):
        try:
            _bytes = self.crypter.decrypt(_token)
        except InvalidToken:
            return b''
        return _bytes

    def unpack(self, buf, b_compress=False,b_encrypt=False,b_base64=False):
        if not b_base64 and b_encrypt:
            buf = base64.urlsafe_b64encode(buf)
        elif b_base64 and not b_encrypt:
            buf = base64.urlsafe_b64decode(buf)
        if b_encrypt:
            buf = self.decrypt(buf)
        if b_compress:
            buf = decompress(buf)
        return buf

    def pack(self, buf, b_compress=False, b_encrypt=False, b_base64=False):
        if b_compress:
            buf = compress(buf, level=self.level)
        if b_encrypt:
            buf = self.encrypt(buf)
        if b_encrypt and (not b_base64):
            buf = base64.urlsafe_b64decode(buf)
        elif (not b_encrypt) and b_base64:
            buf = base64.urlsafe_b64encode(buf)
        return buf

    def unpack_json(self, buf, **kwargs):
        b_dict = self.unpack(buf, **kwargs)
        #print(b_dict[:100])
        s_dict = b_dict.decode('utf-8')        
        dict_ = json.loads(s_dict)
        return dict_

    def pack_json(self, dict_, **kwargs):
        buf = json.dumps(dict_).encode('utf-8')
        return self.pack(buf, **kwargs)

    def pack_array(self, A, dict_={}, **kwargs):
        if not isinstance(A, np.ndarray):
            return None
        bytes_, dtype_, shape_ = array_2_bytes(A)
        dict_['dtype'] = dtype_
        dict_['shape'] = shape_
        b_meta = self.pack_json(dict_, **kwargs)
        len_meta = len(b_meta)
        b_header = len_meta.to_bytes(4, byteorder='big')
        message = b_header+b_meta+bytes_
        return self.pack(message, **kwargs)

    def unpack_array(self, msg_enc, **kwargs):
        if msg_enc is None:
            return None, None
        msg_raw = self.unpack(msg_enc, **kwargs)
        if msg_raw is None or msg_raw == b'FAILED':
            return None, None
        len_header = 4 # byte size of uint32
        b_header = msg_raw[0:len_header]
        len_meta = int.from_bytes(b_header, byteorder='big', signed=False)
        if (len_header+len_meta)>len(msg_raw):
            import struct
            print('Error: header corrupted!')
            print(len(msg_raw), len_header, b_header, struct.unpack("<q", b_header), len_meta)
            return None, None
        b_meta = msg_raw[len_header:len_header+len_meta]
        b_data = msg_raw[len_header+len_meta:]
        dict_ = json.loads(b_meta.decode('utf-8'))
        if ('shape' in dict_) and ('dtype' in dict_):
            dshape = dict_['shape']
            dtype = dict_['dtype']
            if len(b_data)>0:
                A = bytes_2_array(b_data, dtype, dshape)
                return A, dict_
        return None, dict_

    def pack_cmd(self, cmd, b_compress=False, b_encrypt=True, b_base64=True):
        bytes_ = cmd.encode(encoding="utf-8")
        packd_ = self.pack(bytes_, b_compress=b_compress, b_encrypt=b_encrypt, b_base64=b_base64)
        return packd_

    def unpack_cmd(self, packd_, b_compress=False, b_encrypt=True, b_base64=True):
        if packd_ is None:
            return None
        bytes_ = self.unpack(packd_, b_compress=b_compress, b_encrypt=b_encrypt, b_base64=b_base64)
        cmd = bytes_.decode('utf-8')
        return cmd

    def send(self, b_msg):
        try:
            self.stream.socket.send(b_msg)
        except zmq.error.ZMQError as e:
            time.sleep(.1)
            self.count_failed +=1
            if self.count_failed>20:
                #print('reconnect')
                self.stream.reconnect()
                self.count_failed = 0
            #print('send', e)
            return False
        return True

    def recv(self):
        try:
            return self.stream.socket.recv(flags=0, copy=True)
        except zmq.error.ZMQError as e:
            #print('recv', e)
            return None
        
    def query(self, b_msg):
        with self.mutex:
            if self.send(b_msg):
                return self.recv()
            else:
                return None

