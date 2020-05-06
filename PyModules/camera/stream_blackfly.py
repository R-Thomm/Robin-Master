import time
import numpy as np
import os
import PySpin
from .image import streamer_capture
import threading

import socket
import struct

def hex2ip(addr_str):
    addr_long = int(addr_str, 16)
    return socket.inet_ntoa(struct.pack(">L", addr_long))

def hex2mac(addr_str):
    #s = '{0:016x}'.format(addr_long)
    #s = ':'.join(s[i:i + 2] for i in range(0, 16, 2))
    s = addr_str.replace('0x','')
    s = ':'.join(s[i:i + 2].upper() for i in range(0, len(s), 2))
    return s

def print_device_info(cam):
    try:
        result = True
        nodemap = cam.GetTLDeviceNodeMap()
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def get_exposure(cam):
    if cam.ExposureTime.GetAccessMode() == PySpin.RO or cam.ExposureTime.GetAccessMode() == PySpin.RW:
        return float(cam.ExposureTime.ToString())
    else:
        return None

def set_exposure(cam, exposure_time):
    if cam.ExposureTime.GetAccessMode() == PySpin.RW:
        expt_max = cam.ExposureTime.GetMax()
        expt_min = cam.ExposureTime.GetMin()
        print(expt_min, expt_max)
        exposure_time = min(expt_max, exposure_time)
        exposure_time = max(expt_min, exposure_time)
        cam.ExposureTime.SetValue(exposure_time)
        return True
    else:
        return False

def auto_exposure(cam, status):
    if cam.ExposureAuto.GetAccessMode() == PySpin.RW:
        if status:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
        else:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
        return True
    else:
        return False

def auto_gain(cam, status):
    if cam.GainAuto.GetAccessMode() == PySpin.RW and cam.GammaEnable.GetAccessMode() == PySpin.RW:
        if status:
            cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
            cam.GammaEnable.SetValue(True)
        else:
            cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            cam.GammaEnable.SetValue(False)
        return True
    else:
        return False 

def set_gain(cam, gain):
    if cam.Gain.GetAccessMode() == PySpin.RW:
        gain_max = cam.Gain.GetMax()
        gain_min = cam.Gain.GetMin()
        print(gain_min, gain_max)
        gain = min(gain_max, gain)
        gain = max(gain_min, gain)
        cam.Gain.SetValue(gain)
        return True
    else:
        return False

def get_blacklevel(cam):
    if PySpin.IsReadable(cam.BlackLevel):
        return cam.BlackLevel.ToString()
    else:
        return None

def get_serialnumber(cam):
    if cam.TLDevice.DeviceSerialNumber.GetAccessMode() == PySpin.RO:
        return cam.TLDevice.DeviceSerialNumber.ToString()
    else:
        return None

def get_vendorname(cam):
    if PySpin.IsReadable(cam.TLDevice.DeviceVendorName):
        return cam.TLDevice.DeviceVendorName.ToString()
    else:
        return None

def get_displayname(cam):
    if PySpin.IsReadable(cam.TLDevice.DeviceDisplayName):
        return cam.TLDevice.DeviceDisplayName.ToString()
    else:
        return None

def get_modelname(cam):
    if PySpin.IsReadable(cam.TLDevice.DeviceModelName):
        return cam.TLDevice.DeviceModelName.ToString()
    else:
        return None

def set_binning(cam, hbin, vbin):
    if cam.BinningHorizontal.GetAccessMode() == PySpin.RW and cam.BinningVertical.GetAccessMode() == PySpin.RW:
        cam.BinningHorizontal.SetValue(hbin)
        cam.BinningVertical.SetValue(vbin)
        return True
    else:
        return False

def set_roi(cam, x, y, width, height):
    if cam.OffsetX.GetAccessMode() == PySpin.RW and cam.OffsetY.GetAccessMode() == PySpin.RW:
       cam.OffsetX.SetValue(x)
       cam.OffsetY.SetValue(y)
    else:
        return False

    if cam.Width.GetAccessMode() == PySpin.RW and cam.Height.GetAccessMode() == PySpin.RW:
       cam.Width.SetValue(width)
       cam.Height.SetValue(height)
    else:
        return False
    return True

def get_sensor(cam):
    width = int(cam.SensorWidth.ToString())
    height = int(cam.SensorHeight.ToString())
    return height, width

def get_address(cam):
    ip_addr = cam.TLDevice.GevDeviceIPAddress.ToString()
    mac_addr = cam.TLDevice.GevDeviceMACAddress.ToString()
    return hex2ip(ip_addr), hex2mac(mac_addr)

def set_format(cam, ReverseX=False, ReverseY=False, bit=8, rgb=False):
    # set image format. 12 bit ADC. Image format Mono12p. 
    cam.ReverseX.SetValue(ReverseX)
    cam.ReverseY.SetValue(ReverseY)
    if bit > 8:
        image_bit = 16
        cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
    else:
        image_bit = 8
        cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)

    if rgb==0:
        image_rgb = False
        if bit > 8:
            cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
        else:
            cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
    elif rgb>0:
        image_rgb = True
        if rgb==1:
            cam.PixelFormat.SetValue(PySpin.PixelFormat_RGB8Packed)
        else:
            cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)

    max_grayscale = 2**image_bit-1
    return image_bit, max_grayscale, image_rgb

def set_acquisition(cam, acq_mode_single=False):
    if acq_mode_single:
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_SingleFrame)
    else:
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

def capture(cam, num, image_bit, image_rgb):
    result = True
    try:
        cam.BeginAcquisition()
        frames = []; metas = []
        exp = get_exposure(cam)
        #blv = get_blacklevel(cam)
        for i in range(num):
            try:
                ts = time.time()
                image_result = cam.GetNextImage()
                if not image_result.IsIncomplete():
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    depth = image_result.GetBitsPerPixel()
                    if image_rgb:
                        depth = depth/3
                    sat = (2.**depth)-1
                    #print('Grabbed Image %d, width = %d, height = %d, depth = %d' % (i, width, height, depth))
                    '''
                    PySpin.PixelFormat_Mono16, PySpin.PixelFormat_RGB16, PySpin.PixelFormat_BGR8
                    if image_bit > 8: elif image_bit == 8:
                    img_converted = image_result.Convert(format, PySpin.HQ_LINEAR)
                    '''
                    frame = image_result.GetNDArray()
                    if image_rgb:
                        frame = frame.reshape(height, width, 3)
                            
                    image_result.Release()

                    meta = {'saturation':sat, 'timestamp':ts,'exposure':exp, 'shape':[height,width]}
                    frames.append(frame)
                    metas.append(meta)
                    
                else:
                    #print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                    result = False

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                result = False

        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    if not result:
        return None
    else:
        return frames, metas

class streamer_blackfly(object):
    def __init__(self):
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        '''
        # Get current library version
        self.version = self.system.GetLibraryVersion()
        str_ver = '%d.%d.%d.%d' % (self.version.major, self.version.minor, self.version.type, self.version.build)
        print('Library version: %s'%str_ver)
        '''

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()
        self.num_cameras = self.cam_list.GetSize()

        self.cam = None
        print('Number of cameras detected: %d' % self.num_cameras)

        for i, cam in enumerate(self.cam_list):
            print('\tcam id #%i %s %s'%(i, *get_address(cam)))

        self.mutex = threading.Lock()
        
    def __del__(self):
        self.deinit()
        self.cam_list.Clear()
        # Release system instance
        self.system.ReleaseInstance()

    def __setup(self, cam, bit, rgb):
        # Initialize camera
        cam.Init()
        print('Init:')
        name = '%s %s %s'%(get_vendorname(cam), get_modelname(cam), get_serialnumber(cam))
        size = 'Sensor %ix%i'%(get_sensor(cam))
        print('\t%s\n\t%s'%(name,size))

        self.image_bit, self.max_grayscale, self.image_rgb = set_format(cam,bit=bit,rgb=rgb)
        set_acquisition(cam,acq_mode_single=False)

        self.set_full_image()

        self.cam = cam
        return True

    def deinit(self):
        if self.cam is not None:
            # Deinitialize camera
            self.cam.DeInit()
            self.cam = None

    def init(self,index=0, bit=8, rgb=2):
        self.deinit()
        for i in range(self.num_cameras):
            if i==index:
                return self.__setup(self.cam_list.GetByIndex(i), bit=bit,rgb=rgb)
        return False
        
    def get(self, num=1):
        if self.cam is None:
            return None
        else:
            with self.mutex:
                return capture(self.cam, num, self.image_bit, self.image_rgb)
    
    def capture(self):
        ret = self.get()
        if ret is None or len(ret)!=2:
            return None
        else:
            frames, metas = ret
            return frames[0], metas[0]
            
    def set_full_image(self):
        if self.cam is None:
            return False
        else:
            hbin=1; vbin=1; hstart=0; vstart=0
            vend, hend = get_sensor(self.cam)
            return self.set_roi(hbin, vbin, hstart, hend, vstart, vend)
        
    def set_gain(self, gain):
        if self.cam is None:
            return False
        else:
            with self.mutex:
                ret = auto_gain(self.cam, False)
                ret &= set_gain(self.cam, gain)
                return ret

    def set_gain_mode(self, status):
        if self.cam is None:
            return False
        else:
            with self.mutex:
                return auto_gain(self.cam, status)
        
    def set_exposure(self, t):
        if self.cam is None:
            return False
        else:
            with self.mutex:
                ret = auto_exposure(self.cam, False)
                ret &= set_exposure(self.cam, t)
                return ret

    def set_exposure_mode(self, status):
        if self.cam is None:
            return False
        else:
            with self.mutex:
                return auto_exposure(self.cam, status)
    
    def set_binning(self, hbin, vbin):
        if self.cam is None:
            return False
        else:
            with self.mutex:
                return set_binning(self.cam, hbin, vbin)

    def set_roi(self, hbin, vbin, hstart, hend, vstart, vend):
        if self.cam is None:
            return False
        else:
            width = int((hend-hstart)/hbin)
            height = int((vend-vstart)/vbin)
            with self.mutex:
                ret = set_binning(self.cam, hbin, vbin)
                ret &= set_roi(self.cam, hstart, vstart, width, height)
            return ret

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    
    cam = streamer_blackfly()
    if cam.init():
        '''
        ret = cam.capture()
        if ret is not None:
            frame, meta = ret
            print(frame.shape)
            print(meta)
            #plt.imshow(frame)
            #plt.show()
        '''
        cap = streamer_capture(cam)

        ret = cam.capture()
        if ret is not None:
            frame, meta = ret
            cap.save_image(frame, meta)
        
        '''
        print(cap.open())

        frames, metas = cam.get(num=100)
        for f,m in zip(frames,metas):
            cap.append_fast(f,m)
        cap.close()
        '''
        
        '''
        x = threading.Thread(target=cap.video, args=(300,400))
        x.start()

        while True:
            text = input("Command: ")
            if text=='q':
                cap.stop()
                break
            if text:
                ret = False
                par = text.strip().split(' ')
                try:
                    ipar = [int(p) for p in par[1:]]
                except:
                    pass
                try:
                    fpar = [float(p) for p in par[1:]]
                except:
                    pass
                if par[0] == 'roi':
                    hbin, vbin, hstart, hend, vstart, vend = ipar
                    ret = cam.set_roi(hbin, vbin, hstart, hend, vstart, vend)
                elif par[0] == 'bin':
                    hbin, vbin = ipar
                    ret = cam.set_binning(hbin, vbin)
                elif par[0] == 'expt':
                    t = fpar[0]
                    ret = cam.set_exposure(t)
                elif par[0] == 'expm':
                    status = ipar[0]
                    ret = cam.set_exposure_mode(status)
                elif par[0] == 'gain':
                    g = fpar[0]
                    ret = cam.set_gain(g)
                elif par[0] == 'gainm':
                    status = ipar[0]
                    ret = cam.set_gain_mode(status)
                elif par[0] == 'full':
                    ret = cam.set_full_image()
                else:
                    print('unknown')
                print('ret ',ret)
            
        x.join()
        '''
        
