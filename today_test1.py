
import csv

import numpy as np

import serial
import time



# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('com6',9600)
time.sleep(2)
temp=[]

import joblib
clfDT=joblib.load("pipe_DT.joblib")



while(1):
    data  =  ser.readline()  #read the data
    #print(data)
    a=data.decode('latin-1')
    b=a.split(',')
    output=[]
    if float(b[5])==0:
        a=50
    else:
        a=0
    if float(b[4])==0:
        a=a+50
    else:
        a=a

    if float(b[7])==0:
        b1=50
    else:
        b1=0
    if float(b[6])==0:
        b1=b1+50
    else:
        b1=b1

    
    if float(b[1])==0:
        c=50
    else:
        c=0
    if float(b[0])==0:
        c=c+50
    else:
        c=c

    if float(b[2])==0:
        d=50
    else:
        d=0
    if float(b[3])==0:
        d=d+50
    else:
        d=d
    output.insert(0,float(1))
    output.insert(1,float(c))
    output.insert(2,float(d))
    output.insert(3,float(a))
    output.insert(4,float(b1))
    
    out=''
    xc=b[8]
    print(xc)
    for i in range(len(xc)):
        if xc[i]=="\r" or xc[i]=="\n":
            break
        else:
            out=out+xc[i]

    print(output[1:len(output)])
    output=np.array(output)
    preds = clfDT.predict(output.reshape(1,5))
    #out1 = model2.predict(output.reshape(1,5))
    #tar = int(Ans)
    #print(preds)
    print("The target is:",preds)
    ser.write(bytearray(str(preds),'ascii'))
