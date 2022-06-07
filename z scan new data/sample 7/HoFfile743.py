# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz
from math import *

#FILE 442 DOCS
file442_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/z scan new data/sample 7/FILE743.xlsx'
file='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/z scan new data/sample 7/file743ratios.xlsx'

# Load spreadsheet FILE 442
xl = pd.ExcelFile(file442_1storder)
x2 = pd.ExcelFile(file)

# Print the sheet names FILE 442
print(xl.sheet_names)




df4421storder = xl.parse('Sheet2'); df442root3 = xl.parse('Sheet1')

dfskin1 = x2.parse('skin1'); dfcore1 = x2.parse('core1'); dfskinroot3 = x2.parse('skinroot3'); dfcoreroot3 = x2.parse('coreroot3')

#xaxis in degrees [azimuthal angles]
dfdegrees=df4421storder.iloc[:,0]
dfdegrees1 =dfskin1.iloc[:,0]
dfskin12=dfskin1.drop(['DEG'], axis=1)
dfcore12 = dfcore1.drop(['DEG'], axis=1)
dfskinroot32 = dfskinroot3.drop(['DEG'], axis=1)
dfcoreroot32 = dfcoreroot3.drop(['DEG'], axis=1)
df442intensities1storder = df4421storder.drop(['DEG'], axis=1)
df442intensitiesroot3 = df442root3.drop(['DEG'], axis=1)


df2skin1=dfskin1.set_index('DEG')
df2core1 = dfcore1.set_index('DEG')
df2skinroot3 = dfskinroot3.set_index('DEG')
df2coreroot3 = dfcoreroot3.set_index('DEG')


#hermans orientation factor
#core 1
dfhcore1 = np.multiply(np.multiply(np.cos(dfcore1['DEG']*pi/180), np.cos(dfcore1['DEG']*pi/180)), np.sin(dfcore1['DEG']*pi/180))
dfhcore12 = np.sin(dfcore1['DEG']*pi/180)

dfhcore1.squeeze(axis = None)
dfhcore1.transpose()

df2core1int1 = df2core1.copy()
func = lambda x: np.asarray(x) * np.asarray(dfhcore1)
dfexcess = df2core1int1.apply(func)

funcnew = lambda x: np.asarray(x) * np.asarray(dfhcore12)
dfexcess1 = df2core1int1.apply(funcnew)

n=len(df442intensities1storder.columns)

frame_array442=[]
for i in range(1, n+1):
    frame=i
    frame_array442.append(frame)

areatrapc1num = []
areatrapc1den = []
# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = dfexcess.iloc[:,i]
    z = dfexcess1.iloc[:,i]

 
    x = dfdegrees
  
    areac1num = trapz(y, x)
    areac1den = trapz(z,x)

    areatrapc1num.append(areac1num)
    areatrapc1den.append(areac1den)


print("areatrap4421storder_array =", areatrapc1num)
print("areatrap442root3_array =", areatrapc1den)
#finally perform integrals
ratiointc1 = np.divide(areatrapc1num, areatrapc1den)

valuesc1 = (3*ratiointc1 - 1)/2

valuesc1 = np.where((valuesc1 < -1), nan, valuesc1)
valuesc1 = np.where((valuesc1 > 1), nan, valuesc1)

dfhcore3 = np.multiply(np.multiply(np.cos(dfcoreroot3['DEG']*pi/180), np.cos(dfcoreroot3['DEG']*pi/180)), np.sin(dfcoreroot3['DEG']*pi/180))
dfhcore32 = np.sin(dfcoreroot3['DEG']*pi/180)

dfhcore3.squeeze(axis = None)
dfhcore3.transpose()

df2core3int1 = df2coreroot3.copy()
func1 = lambda x: np.asarray(x) * np.asarray(dfhcore3)
dfexcess2 = df2core3int1.apply(func1)

funcnew1 = lambda x: np.asarray(x) * np.asarray(dfhcore32)
dfexcess3 = df2core3int1.apply(funcnew1)

n=len(df442intensities1storder.columns)

frame_array442=[]
for i in range(1, n+1):
    frame=i
    frame_array442.append(frame)

areatrapc3num = []
areatrapc3den = []
# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = dfexcess2.iloc[:,i]
    z = dfexcess3.iloc[:,i]

 
    x = dfdegrees
  
    areac3num = trapz(y, x)
    areac3den = trapz(z,x)

    areatrapc3num.append(areac3num)
    areatrapc3den.append(areac3den)


print("areatrap4421storder_array =", areatrapc3num)
print("areatrap442root3_array =", areatrapc3den)
#finally perform integrals
ratiointc3 = np.divide(areatrapc3num, areatrapc3den)

valuesc3 = (3*ratiointc3 - 1)/2


valuesc3 = np.where((valuesc3 < -1), nan, valuesc3)
valuesc3 = np.where((valuesc3 > 1), nan, valuesc3)
#valuesc3s = valuesc3.astype(float)
#valuesc3s[(valuesc3s < -1) & ( valuesc3s > 1)] = np.nan
#valuesc3s


fig1, ax1 = plt.subplots(figsize=(12,8))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.scatter(frame_array442, valuesc1, marker=".", label='HoF 1st order core')
l2 =  ax1.scatter(frame_array442, valuesc3, marker=".", label='HoF root 3 core')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
#ax1.legend()
ax1.legend()
ax1.set_ylabel('HoF')
ax1.set_xlabel('Frame number')
plt.grid()
ax1.set_title('FILE 743 sample 7 thickness 0.55-0.73 mm')
plt.show()



#hermans orientation factor
#core 1
dfhskin1 = np.multiply(np.multiply(np.cos(dfskin1['DEG']*pi/180), np.cos(dfskin1['DEG']*pi/180)), np.sin(dfskin1['DEG']*pi/180))
dfhcore12 = np.sin(dfcore1['DEG']*pi/180)

dfhskin1.squeeze(axis = None)
dfhskin1.transpose()

df2skin1int1 = df2skin1.copy()
funcs = lambda x: np.asarray(x) * np.asarray(dfhskin1)
dfexcesss = df2skin1int1.apply(func)

funcsnew = lambda x: np.asarray(x) * np.asarray(dfhcore12)
dfexcesss1 = df2skin1int1.apply(funcsnew)

n=len(df442intensities1storder.columns)

frame_array442=[]
for i in range(1, n+1):
    frame=i
    frame_array442.append(frame)

areatraps1num = []
areatraps1den = []
# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = dfexcesss.iloc[:,i]
    z = dfexcesss1.iloc[:,i]

 
    x = dfdegrees
  
    areas1num = trapz(y, x)
    areas1den = trapz(z,x)

    areatraps1num.append(areas1num)
    areatraps1den.append(areas1den)


print("areatrap4421storder_array =", areatraps1num)
print("areatrap442root3_array =", areatraps1den)
#finally perform integrals
ratioints1 = np.divide(areatraps1num, areatraps1den)

valuess1 = (3*ratioints1 - 1)/2

valuess1 = np.where((valuess1 < -1), nan, valuess1)
valuess1 = np.where((valuess1 > 1), nan, valuess1)


dfhskin3 = np.multiply(np.multiply(np.cos(dfskinroot3['DEG']*pi/180), np.cos(dfskinroot3['DEG']*pi/180)), np.sin(dfskinroot3['DEG']*pi/180))
dfhskin32 = np.sin(dfskinroot3['DEG']*pi/180)

dfhskin3.squeeze(axis = None)
dfhskin3.transpose()

df2skin3int1 = df2skinroot3.copy()
funcs1 = lambda x: np.asarray(x) * np.asarray(dfhskin3)
dfexcesss2 = df2skin3int1.apply(funcs1)

funcsnew1 = lambda x: np.asarray(x) * np.asarray(dfhskin32)
dfexcesss3 = df2skin3int1.apply(funcnew1)

n=len(df442intensities1storder.columns)

frame_array442=[]
for i in range(1, n+1):
    frame=i
    frame_array442.append(frame)

areatraps3num = []
areatraps3den = []
# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = dfexcesss2.iloc[:,i]
    z = dfexcesss3.iloc[:,i]

 
    x = dfdegrees
  
    areas3num = trapz(y, x)
    areas3den = trapz(z,x)

    areatraps3num.append(areas3num)
    areatraps3den.append(areas3den)


print("areatrap4421storder_array =", areatraps3num)
print("areatrap442root3_array =", areatraps3den)
#finally perform integrals
ratioints3 = np.divide(areatraps3num, areatraps3den)

valuess3 = (3*ratioints3 - 1)/2

valuess3 = np.where((valuess3 < -1), nan, valuess3)
valuess3 = np.where((valuess3 > 1), nan, valuess3)

fig1, ax1 = plt.subplots(figsize=(12,8))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.scatter(frame_array442, valuess1, marker=".", label='HoF 1st order skin')
l2 =  ax1.scatter(frame_array442, valuess3, marker=".", label='HoF root 3 skin')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
#ax1.legend()
ax1.legend()
ax1.set_ylabel('HoF')
ax1.set_xlabel('Frame number')
plt.grid()
ax1.set_title('FILE 743 sample 7 thickness 0.55-0.73 mm')
plt.show()


fig1, ax1 = plt.subplots(figsize=(12,8))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.scatter(frame_array442, valuess1, marker=".", label='HoF 1st order skin')
l2 =  ax1.scatter(frame_array442, valuesc1, marker=".", label='HoF 1st order core')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
#ax1.legend()
ax1.legend()
ax1.set_ylabel('HoF')
ax1.set_xlabel('Frame number')
plt.grid()
ax1.set_title('FILE 743 sample 7 thickness 0.55-0.73 mm')
plt.show()

fig1, ax1 = plt.subplots(figsize=(12,8))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.scatter(frame_array442, valuesc3, marker=".", label='HoF root 3 core')
l2 =  ax1.scatter(frame_array442, valuess3, marker=".", label='HoF root 3 skin')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
#ax1.legend()
ax1.legend()
ax1.set_ylabel('HoF')
ax1.set_xlabel('Frame number')
plt.grid()
ax1.set_title('FILE 743 sample 7 thickness 0.55-0.73 mm')
plt.show()
