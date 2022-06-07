# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz
from math import *

#FILE 442 DOCS
file442_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE610SAMPLE14HOF.xlsx'


# Load spreadsheet FILE 442
xl = pd.ExcelFile(file442_1storder)


# Print the sheet names FILE 442
print(xl.sheet_names)




df4421storder = xl.parse('Sheet2'); df442root3 = xl.parse('Sheet1')



#xaxis in degrees [azimuthal angles]
dfdegrees=df4421storder.iloc[:,0]

df442intensities1storder = df4421storder.drop(['DEG'], axis=1)
df442intensitiesroot3 = df442root3.drop(['DEG'], axis=1)

# The y values.  A numpy array is used here,
# but a python list could also be used.
#area_array=[]
n=len(df442intensities1storder.columns)
#y = dfintensities.iloc[:,80]
#x = dfdegrees
# Compute the area using the composite trapezoidal rule.
#area = trapz(y, x)
#print("area =", area)
frame_array442=[]
for i in range(1, n+1):
    frame=i
    frame_array442.append(frame)

areatrap4421storder_array=[]
areatrap442root3_array=[]
areacore1 = []
areaskin1 = []
areacoreroot3 = []
areaskinroot3 = []
# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = df442intensities1storder.iloc[:,i]
    z = df442intensitiesroot3.iloc[:,i]

 
    x = dfdegrees
   
    area4421storder = trapz(y, x)
    area442root3 = trapz(z,x)
  

    areatrap4421storder_array.append(area4421storder)
    areatrap442root3_array.append(area442root3)


print("areatrap4421storder_array =", areatrap4421storder_array)
print("areatrap442root3_array =", areatrap442root3_array)


#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig1, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array442, areatrap4421storder_array, marker="None", label='1st order peaks')
l2 =  ax1.plot(frame_array442, areatrap442root3_array, marker="None", label= 'root 3 order peaks')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 635')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 442
df2442root3 = df442root3.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
linesnew = df2442root3.plot.line(legend=False, grid = True)
plt.show()


#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 442
df24421storder = df4421storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
#lines1 = df24421storder.iloc[:,160].plot.line(legend=False, grid = True)
fig2, ax2 = plt.subplots(figsize=(10,6))


#PLOT ROOT 3 AND 2ND ORDER PEAKS FOR FILE 442
plt.plot(frame_array442, areatrap442root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
#plt.xticks(np.arange(0, 100, 5))
plt.title('FILE 708')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()
    




#EXTENDED WORK FOR DATA ANALYSIS [PEAK RATIOS AND SKIN AND CORE FRACTION METHODOLOGY]

#DIVIDE NUMPY ARRAYS 1ST ORDER AND ROOT 3 IN SAME ORDER AS THEY APPEAR AT THE TOP OF THE FILE

file442ratio1 = np.divide(areatrap4421storder_array, areatrap442root3_array)


#PUTTING THEM ALL TOGETHER

#fig10, axs = plt.subplots(nrows = 6, ncols = 2, constrained_layout = True, figsize=(20,20))
#(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8 ), (ax9, ax10), (ax11, ax12) = axs
#fig10.suptitle('FILES 442, 499, 505 & 472')


plt.plot(frame_array442, file442ratio1, color='g', marker='None', ls='-',label='FILE 708')

plt.xlabel('Frame')
plt.ylabel('Intensity ratio')
plt.legend(loc = 'upper right')
plt.grid()


#fig10.delaxes(axs[5,1]) #The indexing is zero-based here
plt.show()






#polynomial regression
B = (len(df4421storder.columns)-1)

print(range(1, B+1))
y_data = pd.DataFrame() 
ydata = pd.DataFrame()
coefficients = pd.DataFrame()
poly = pd.DataFrame()
new_y = pd.DataFrame()
for i in range(1, B+1):
    print(df4421storder[i])
    x_data, y_data[i] = (df4421storder['DEG'], df4421storder[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficients[i] = np.polyfit(x_data, y_data[i], 120)
    poly = np.poly1d(coefficients[i])

    new_y[i] = poly(x_data)

    plt.figure(figsize=(8,5))
#y = sigmoid(xdata, *popt)


print(new_y[:])

plt.figure()
plt.plot(x_data, y_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('2theta')
plt.show()

plt.figure()
plt.plot(df4421storder['DEG'], df24421storder[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('2theta')
plt.show()

plt.figure()
plt.plot(x_data, new_y[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('2theta')
plt.show()

C = (len(df442root3.columns)-1)

print(range(1, B+1))
z_data = pd.DataFrame() 
zdata = pd.DataFrame()
coefficientsnew = pd.DataFrame()
polyx = pd.DataFrame()
new_z = pd.DataFrame()
for i in range(1, B+1):
    print(df442root3[i])
    x_data, z_data[i] = (df442root3['DEG'], df442root3[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew[i] = np.polyfit(x_data, z_data[i], 120)
    polyx = np.poly1d(coefficientsnew[i])

    new_z[i] = polyx(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, z_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()

plt.figure()
plt.plot(df442root3['DEG'], df2442root3[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()

plt.figure()
plt.plot(x_data, new_z[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()

dfc1 = (df24421storder-20000)
dfc1[dfc1 < 0]=0
linesnew = dfc1.iloc[0:1000,:].plot.line(legend=False, grid = True)
plt.show()

dfc3 = (df2442root3-2000)
dfc3[dfc3 < 0]=0
linesnew = dfc3.iloc[0:1000,:].plot.line(legend=False, grid = True)
plt.show()

#hermans orientation factor
#core 1
dfhcore1 = np.multiply(np.multiply(np.cos(df4421storder['DEG']*pi/180), np.cos(df4421storder['DEG']*pi/180)), np.sin(df4421storder['DEG']*pi/180))
dfhcore12 = np.sin(df4421storder['DEG']*pi/180)

dfhcore1.squeeze(axis = None)
dfhcore1.transpose()

df2core1int1 = dfc1.copy()
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

dfhcore3 = np.multiply(np.multiply(np.cos(df442root3['DEG']*pi/180), np.cos(df442root3['DEG']*pi/180)), np.sin(df442root3['DEG']*pi/180))
dfhcore32 = np.sin(df442root3['DEG']*pi/180)

dfhcore3.squeeze(axis = None)
dfhcore3.transpose()

df2core3int1 = dfc3.copy()
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
l1 =  ax1.scatter(frame_array442, valuesc1, marker=".", label='HoF 1st order')
l2 =  ax1.scatter(frame_array442, valuesc3, marker=".", label='HoF root 3')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
#ax1.legend()
ax1.set_ylabel('HoF')
ax1.set_xlabel('Frame number')
#plt.grid()
ax1.set_title('FILE 577 - sample 7 perpendicular cut')
plt.show()