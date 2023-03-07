from pydoc import apropos
from tempfile import TemporaryDirectory
import os
import numpy as np
import matplotlib.pyplot as plt


#Importing data

def Dataimport(filename,flow_rate=0.5):
  path= os.getcwd()
  path = path + "/" + str(filename)
  data = np.genfromtxt(path, skip_header=2, delimiter=',')
  time = data[:,0]
  data = data[:,1::2]
  volume= time*flow_rate
  return data, time, volume

data, time, volume = Dataimport(filename,flow_rate=0.5)



#Plotting data """

def FSEC4deg():
  plt.figure()
  plt.plot(volume, data[:,0],'k', label='CON_011')
  plt.plot(volume, data[:,12], label='CON_013')
  plt.plot(volume, data[:,23], label='CON_014')
  plt.plot(volume, data[:,34], label='CON_019')
  plt.plot(volume, data[:,45], label='CON_021')
  plt.plot(volume, data[:,56], label='CON_023')
  plt.plot(volume, data[:,67], label='CON_027')
  plt.plot(volume, data[:,78], label='CON_030')
  plt.ylabel('Fluorescence Intensity, Ex= 488')
  plt.title('all constructs - 4°C')
  plt.xlim(0,3)
  plt.legend()
  plt.grid()


#MANUAL INPUTS:

#to select range for max searching
volume[527:560]
no_con = 8 #number of different construct
no_temp = 11 # number of different temperatures
temp = np.array((4.0,32.7,40.7,46.2,51.8,57.0,63.0,68.5,73.7,79.2,84.8)) #manually input the used temperatures
con_label =  ['CON_011', 'CON_013', 'CON_014', 'CON_019', 'CON_021', 'CON_023', 'CON_027', 'CON_030']

#This function returns the maximum, within the selected range, for a number of constructs and temperatures
# data output is maxs[temperatures,constructs]

def max_generator(no_temp, no_con):
  maxs = np.zeros((no_temp,no_con))
  for x in range(no_con):
    for i in range(no_temp):
      maxs[i,x]= data_e[527:560,(x*no_temp+i)].max()   
  return maxs

      

def FSEC_TS(temp,maxs,no_con,con_label):
  plt.figure()
  for i in range(no_con):
    plt.plot(temp, maxs[:,i], label=str(con_label[i]))
  plt.ylabel('Fluorescence Intensity, Ex= 488 at max peak')
  plt.xlabel("Temp °C")
  plt.title('FSEC_TS')
  plt.xlim(0,90)
  plt.legend()
  plt.grid()

def FSEC_TS_norm(temp,maxs,no_con,con_label):
  plt.figure()
  for i in range(no_con):
    plt.plot(temp, maxs[:,i]/maxs[0,i], label=str(con_label[i]))
  plt.ylabel('Normalized Fluorescence Intensity, Ex= 488 at max peak')
  plt.xlabel("Temp °C")
  plt.title('FSEC_TS')
  plt.xlim(0,90)
  plt.legend()
  plt.grid()




# define sigmoidal fit

from scipy.optimize import curve_fit
def sigmoid(x, L ,x0, k, b):
  y = L / (1 + np.exp(-k*(x-x0))) + b
  return y

#Function to fit the sigmoid curves to the data, normalized or not. Returns the parameters optimized (popt = L, x0,k,b ) for one single curve eache time. x0 corresponds to the Tm  and b to the baseline offset

def FitSigmoid(data, Normalized = True):
  if Normalized == True:
    data= data/data[0]
    p0 = [max(data), np.median(temp),-1,min(data)]
    popt, pcov = curve_fit(sigmoid, temp, data,p0,bounds= ((0,0,-np.inf,0),(1,100,np.inf,1)), method='dogbox')
    x = np.linspace(0,100,1000)
    fit =sigmoid(x, *popt)
    return popt, x, fit    
  else:
    p0 = [max(data), np.median(temp),-1,min(data)]
    popt, pcov = curve_fit(sigmoid, temp, data,p0,bounds= ((0,0,-np.inf,0),(np.inf,100,np.inf,np.inf)), method='dogbox')
    x = np.linspace(0,100,1000)
    fit =sigmoid(x, *popt)
    return popt, x, fit


def FSEC_TS_fits(Normalized = True):
  if Normalized ==True:
    FSEC_TS_norm()
    for i in list(range(no_con)):
      popt,x,fit = FitSigmoid(maxs[:,i], Normalized=True)
      plt.plot(x,fit,'k:')
      print("Tm of {} = ". format(con_label[i]) + str(popt[1])) 
  else:
    FSEC_TS(temp,maxs,no_con,con_label)
    for i in list(range(no_con)):
      popt,x,fit = FitSigmoid(maxs[:,i], Normalized=False)
      plt.plot(x,fit,'k:')
      print("Tm of {} = ". format(con_label[i]) + str(popt[1])) 

  





  #plotting and fitting only the last 5 temperatures:

  