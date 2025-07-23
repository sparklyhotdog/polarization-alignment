# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:33:37 2024

@author: yns19
"""

import pyvisa
import sys, time

class HP8164A():
    
    def __init__(self, name='ASRL9::INSTR'):
        rm=pyvisa.ResourceManager()
        deviceList=rm.list_resources()
        FoundDevice=False
        print(deviceList)
        #looking through the device list
        for i in deviceList:
            #print(i)
            if name in i:
                device=i
                print("Tunable laser found, ID:", device)
                FoundDevice=True
                
        if FoundDevice:
            try:
                self.tLaser=rm.open_resource(device, baud_rate=9600,
                                             data_bits=8,
                                             parity=pyvisa.constants.Parity.none,
                                             stop_bits=pyvisa.constants.StopBits.one,
                                             flow_control=pyvisa.constants.VI_ASRL_FLOW_NONE)
                self.tLaser.timeout=1000
                self.tLaser.read_termination='\n'
                print("tunable laser connected.")
            except OSError:
                print('Cannot open device.')
        else:
            print("Device not found.")
            sys.exit()
    
    #Query current wavelength
    def getWavelength(self):
        try:
            out=self.tLaser.query("SOUR1:WAV?")
        except:
            out='ERR\n'
        #out=self.synthesizer.read()
        #print(out)
        return out.strip('\n')    
    
    '''
    Set current wavelength, input wavelength in unit of micro-meters
    '''
    def setWavelength(self, wave):
        self.tLaser.write('SOUR1:WAV % 6fe-6'%(wave))
        
    def idn(self):
        print(self.tLaser.query("*IDN?"))
    
    def close(self):
        self.tLaser.close()                    


if __name__ == '__main__':
    tl=HP8164A(name='ASRL9::INSTR')
    tl.idn()
    print(tl.getWavelength())
    tl.setWavelength(1.54)

        
    tl.close()

    pass
    