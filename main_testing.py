import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')

current = input("enter the current status")
#print("current status:" +current+ ".")
print(f"current status: {current}")

signalA = input("enter the signal A")
print(f"signalA: {signalA}")

signalB = input("enter the signal B")
print(f"signalB: {signalB}")

signalC = input("enter the signal C")
print(f"signalC: {signalC}")

signalD = input("enter the signal D")
print(f"signalD: {signalD}")


import pickle


import joblib
clfDT=joblib.load("pipe_DT.joblib")

res = [[float(current),float(signalA),float(signalB),float(signalC),float(signalD)]]
res=np.array(res)
preds = clfDT.predict(res.reshape(1,5))

tar = int(preds)
print("The target is:",tar)
