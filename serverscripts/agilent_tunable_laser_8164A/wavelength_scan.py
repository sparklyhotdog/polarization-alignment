# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:16:14 2024

@author: yns19
"""

import psy201_control as pc
import tunable_laser as tl
import numpy as np
import time, os
import datetime


basefn = 'polarization-tunable-wavelength'
dr     = 'C:\\Users\\yns19\\OneDrive - NIST\\yicheng-lab-book\\POTDR\\polarization-correlation'  
tim  = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
fn   = dr + os.sep + basefn + '_' + tim + '.dat'
fn2 = dr + os.sep + '6kmPML-L.dat'

tuna=tl.HP8164A(name='GPIB1::20::INSTR')
pola=pc.psy201(name='GPIB0::5::INSTR')

tuna.idn()
pola.idn()

print(pola.getSOP())
print(tuna.getWavelength())

tuna.setWavelength(1.525)
time.sleep(15)
with open(fn2, "a") as f:
    waveRange=np.linspace(1.525, 1.575, 501)
    tStart=datetime.datetime.now()
    for i in waveRange:
        out= []
        tuna.setWavelength(i)
        #pola.setWavelength(int(round(i,4)*1000))
        time.sleep(0.3)
        pol=pola.getSOP()
        #a=p1.idn()
        #b=p2.idn()
        print('take'+str(i))
        bla1=pol.split(',')
        tElapsed=datetime.datetime.now()-tStart
        out=np.append([tElapsed.total_seconds(), round(i,5)], bla1)
        np.savetxt(f, out, newline=" ", fmt="%s")
        f.write("\n")
        '''
        print('take'+str(i))
        bla2=[float(i) for i in bla1]
        #time.sleep(1)
        '''

f.close()
pola.close()
tuna.close()
