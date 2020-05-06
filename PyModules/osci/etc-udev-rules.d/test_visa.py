
import usb

devices = usb.core.find(find_all=True)
#devices = usb.core.find(idVendor=0x1ab1,find_all=True)
for dev in devices:
    #devices.set_configuration()
    try:
        print(dev.serial_number)
        print(dev.idVendor)
    except ValueError as e:
        print('USB Error: ',e)


import visa
rm = visa.ResourceManager()
try:
    print(rm.list_resources(query=u'?*::INSTR'))
except ValueError as e:
    print(e)
