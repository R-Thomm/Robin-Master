import socket
#import os

class IPC_client:
    def __init__(self, server_address, buffer_size=16384): #32768
        self.buffer_size =  buffer_size
        self.server_address = server_address
        print('socket connecting to %s' % self.server_address)
        self.connect()
        
    def __del__(self):
        self.sock.close()
        #os.unlink(file)
        print('socket closed: %s'%self.server_address)

    def connect(self):
        # Create a UDS socket
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.server_address)

    def reconnect(self):
        print('socket reconnect to %s' % self.server_address)
        self.sock.close()
        self.connect()
        
    def _read(self, buffer_size):
        data = self.sock.recv(buffer_size)
        #print('%i,%.2f'%(len(data), len(data)/buffer_size))
        return data, len(data)==buffer_size

    def send(self, message):
        data = message.encode()
        ret = None
        try:
            ret = self.sock.sendall(data)
        except socket.error as e:
            print('socket error: ', e)
            self.reconnect()
            ret = self.sock.sendall(data)
        except IOError as e:
            if e.errno == errno.EPIPE:
                raise Exception('IOError (EPIPE): broken pipe')
            else:
                raise Exception('IOError other: ', e)
        return ret==None
    
    def receive(self):
        data, overflow = self._read(self.buffer_size)
        while(overflow):
            msg, overflow = self._read(self.buffer_size)
            data += msg
        
        #print('received %i bytes: "%s..."' % (len(data),data[:10]))
        return data.decode()
        
    def query(self,message):
        if not self.send(message):
            return None
        return self.receive()

