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
    z1 = dfskin12.iloc[:,i]
    z2 = dfcore12.iloc[:,i]
    z3 = dfskinroot32.iloc[:,i]
    z4 = dfcoreroot32.iloc[:,i]
 
    x = dfdegrees
    z5 = dfdegrees1
    area4421storder = trapz(y, x)
    area442root3 = trapz(z,x)
    skin1trap=trapz(z1, z5)
    core1trap=trapz(z2,z5)
    skinroottrap = trapz(z3, z5)
    coreroottrap = trapz(z4,z5)

    areatrap4421storder_array.append(area4421storder)
    areatrap442root3_array.append(area442root3)
    areacore1.append(core1trap)
    areaskin1.append(skin1trap)
    areacoreroot3.append(coreroottrap)
    areaskinroot3.append(skinroottrap)

print("areatrap4421storder_array =", areatrap4421storder_array)
print("areatrap442root3_array =", areatrap442root3_array)


#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig1, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array442, areatrap4421storder_array, 'r.', marker=".", label='1st order peaks')
l2 =  ax1.plot(frame_array442, areatrap442root3_array, 'b.', marker=".", label= 'root 3 order peaks')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend(loc = 'lower right')
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 743 - sample 7 thickness 0.55-0.73 mm')
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
lines1 = df24421storder.iloc[:,160].plot.line(legend=False, grid = True)
fig2, ax2 = plt.subplots(figsize=(10,6))


#PLOT ROOT 3 AND 2ND ORDER PEAKS FOR FILE 442
plt.plot(frame_array442, areatrap442root3_array, 'r.', color='r', marker='.', label='root 3 peaks')
#plt.xticks(np.arange(0, 100, 5))
plt.title('FILE 743 sample 7 thickness 0.55-0.73 mm')
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


plt.plot(frame_array442, file442ratio1, 'g.', color='g', marker='.', label='FILE 708')
plt.title('FILE 743 sample 7 1st order root 3 ratio')
plt.xlabel('Frame')
plt.ylabel('Intensity ratio')
#plt.legend(loc = 'upper right')
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


#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig4, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array442, areacore1, 'r.',  marker=".", label='1st order core')
l2 =  ax1.plot(frame_array442, areaskin1, 'b.',  marker=".", label= '1st order skin')
l3 =  ax1.plot(frame_array442, areacoreroot3, 'g.',  marker=".", label= 'root 3 core')
l4 =  ax1.plot(frame_array442, areaskinroot3, 'k.',  marker=".", label= 'root 3 skin')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend(loc ='lower right')
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 743 sample 7 thickness 0.55-0.73 mm')
plt.show()


df2skin1=dfskin1.set_index('DEG')
df2core1 = dfcore1.set_index('DEG')
df2skinroot3 = dfskinroot3.set_index('DEG')
df2coreroot3 = dfcoreroot3.set_index('DEG')


linesnew1 = df2skin1.plot.line(legend=False, grid = True, title = 'skin 1st order')


linesnew2 = df2core1.plot.line(legend=False, grid = True, title = 'core 1st order')

linesnew3 = df2skinroot3.plot.line(legend=False, grid = True, title = 'skin root 3')

linesnew4 = df2coreroot3.plot.line(legend=False, grid = True, title = 'core root 3')
#find max 1st order for core starts at index 12





ratioskincore1 = np.divide(areaskin1, areaskinroot3)
ratioskincoreroot3 = np.divide(areacore1, areacoreroot3)


fig10, axs = plt.subplots(nrows = 1, ncols = 2, constrained_layout = True, figsize=(24,12))
(ax1, ax2) = axs
fig10.suptitle('FILE 743 sample 7 thickness 0.55-0.73 mm')

ax1.plot(frame_array442, ratioskincoreroot3, 'b.',  color='b', marker='.', label='skin/core root 3')
ax1.set_title('RATIO FOR ROOT 3')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Total Azimuthal Integrated Intensity ratio')
#ax2.legend(loc = 'upper right')
ax1.grid()

ax2.plot(frame_array442, ratioskincore1, 'r.',  color='r', marker='.', label='skin/core 1st order')
ax2.set_title('RATIO FOR 1ST ORDER')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Total Azimuthal Integrated Intensity ratio')
#ax2.legend(loc = 'upper right')
ax2.grid()

#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()


#method do it for 1 peak in specified ranges

#print(dfmaxcorehalfmaximum[1])
#180 degrees skin root 3
dfsr3= (df2skinroot3.iloc[650:720,:] - 65)
dfsr3[dfsr3 < 0] = 0
(df2skinroot3.iloc[650:720,:] - 65).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'Root 3 peaks for skin at 180 degrees ' )
plt.show()


#odegrees skin

#all skin root 3
dfsr3clean = (df2skinroot3 - 65)
dfsr3clean[dfsr3clean < 0]=0
dfsr3clean.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'Root 3 peaks for skin overall ' )
plt.show()

dfsr32 = dfsr3clean.iloc[0:40,:]
dfsr32.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'Root 3 peaks for skin at 0 degrees ' )

dfsr33 = dfsr3clean.iloc[1350:1400,:]
dfsr33.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'Root 3 peaks for skin at 0 degrees ' )
#polynomial fit for skin root3
d = (len(dfsr3.columns)-1)

print(range(1, d+1))
u_data = pd.DataFrame() 
udata = pd.DataFrame()
coefficientsnew1 = pd.DataFrame()
polyx1 = pd.DataFrame()
new_z1 = pd.DataFrame()
for i in range(1, d+1):

    x_data, u_data[i] = (dfskinroot3['DEG'].loc[650:719], dfsr3[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew1[i] = np.polyfit(x_data, u_data[i], 120)
    polyx1 = np.poly1d(coefficientsnew1[i])

    new_z1[i] = polyx1(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, u_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z1[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()


#polynomial fit for entire thing
print(range(1, d+1))

m_data = pd.DataFrame() 
mdata = pd.DataFrame()
coefficientsnew2 = pd.DataFrame()
polyx2 = pd.DataFrame()
new_z2 = pd.DataFrame()
for i in range(1, d+1):

    x_data, m_data[i] = (dfskinroot3['DEG'], dfsr3clean[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew2[i] = np.polyfit(x_data, m_data[i], 120)
    polyx2 = np.poly1d(coefficientsnew2[i])

    new_z2[i] = polyx2(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, m_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z2[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()


#skin 1st order
dfs1 = (df2skin1-170)
dfs1[dfs1 < 0]=0
dfsr1 = dfs1.iloc[590:770, :]



dfs1[dfs1 < 0]=0
dfs1.iloc[590:770,:].plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = '1st order peaks for skin ' )
plt.show()

dfsr12 = dfs1.iloc[0:140,:]
dfsr12.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = '1st order peaks for skin at 0 degrees ' )

dfsr13 = dfs1.iloc[1250:1400,:]
dfsr13.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = '1st order peaks for skin at 0 degrees ' )

e = (len(dfs1.columns)-1)

#polynomial fit for skin 1st order all

l_data = pd.DataFrame() 
ldata = pd.DataFrame()
coefficientsnew3 = pd.DataFrame()
polyx3 = pd.DataFrame()
new_z3 = pd.DataFrame()
for i in range(1, e+1):

    x_data, l_data[i] = (dfskin1['DEG'], dfs1[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew3[i] = np.polyfit(x_data, l_data[i], 120)
    polyx3 = np.poly1d(coefficientsnew3[i])

    new_z3[i] = polyx3(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, l_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z3[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()

#polynomial fit for skin 1st order specified range


n_data = pd.DataFrame() 
ndata = pd.DataFrame()
coefficientsnew4 = pd.DataFrame()
polyx4 = pd.DataFrame()
new_z4 = pd.DataFrame()
for i in range(1, e+1):

    x_data, n_data[i] = (dfskin1['DEG'].loc[590:769], dfsr1[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew4[i] = np.polyfit(x_data, n_data[i], 120)
    polyx4 = np.poly1d(coefficientsnew4[i])

    new_z4[i] = polyx4(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, n_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z4[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()

#method do it for 1 peak in specified ranges
#get angles for maxima

fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('Angles at which peak maxima occur for root 3 and 1st order skin orientation')
dfskin3maxangle = dfsr3.idxmax()
dfskin1maxangle = dfsr1.idxmax()
x = np.arange(1,(len(dfskin1maxangle) + 1),1)
dfskin1maxanglenew = pd.DataFrame(dfskin1maxangle).to_numpy()
dfskin3maxanglenew = pd.DataFrame(dfskin3maxangle).to_numpy()
l1 = ax1.plot(x, dfskin3maxanglenew, 'r.', label = 'skin root 3 max peak angle')
l2 = ax1.plot(x, dfskin1maxanglenew, 'b.',  label = 'skin 1st order max peak angle')
ax1.legend()
ax1.set_ylabel('Azimuthal angle')
ax1.set_xlabel('Frame number')
plt.show()


#for core

dfcr3= (df2coreroot3.iloc[320:410,:] - 80)
dfcr3[dfcr3 < 0] = 0
(df2coreroot3.iloc[310:420,:] - 80).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'Root 3 peaks for core ' )
plt.show()



#all core root 3
dfcr3clean = (df2coreroot3 - 80)
dfcr3clean[dfcr3clean < 0]=0
dfcr3clean.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'Root 3 peaks for core overall ' )
plt.show()

dfcr32 = dfcr3clean.iloc[1010:1110, :]
dfcr32.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'Root 3 peaks for core' )
plt.show()
#polynomial fit for core root3
g = (len(dfcr3.columns)-1)


o_data = pd.DataFrame() 
odata = pd.DataFrame()
coefficientsnew5 = pd.DataFrame()
polyx5 = pd.DataFrame()
new_z5 = pd.DataFrame()
for i in range(1, g+1):

    x_data, o_data[i] = (dfcoreroot3['DEG'].loc[320:409], dfcr3[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew5[i] = np.polyfit(x_data, o_data[i], 120)
    polyx5 = np.poly1d(coefficientsnew5[i])

    new_z5[i] = polyx5(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, o_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z5[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()

#polynomial fit for core root 3 order all
p_data = pd.DataFrame() 
pdata = pd.DataFrame()
coefficientsnew6 = pd.DataFrame()
polyx6 = pd.DataFrame()
new_z6 = pd.DataFrame()
for i in range(1, g+1):

    x_data, p_data[i] = (dfcoreroot3['DEG'], dfcr3clean[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew6[i] = np.polyfit(x_data, p_data[i], 120)
    polyx6 = np.poly1d(coefficientsnew6[i])

    new_z6[i] = polyx6(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, p_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z6[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()


#core 1st order
dfc1 = (df2core1-170)
dfc1[dfc1 < 0]=0
dfcr1 = dfc1.iloc[240:465, :]

#core1st order 270 deg
dfcr12 = dfc1.iloc[1010:1130, :]
dfcr12.plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = '1st order peaks for core ' )

dfc1.iloc[:, :].plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = '1st order peaks for core ' )
plt.show()

dfc1[dfc1 < 0]=0
dfc1.iloc[240:465,:].plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = '1st order peaks for core ' )
plt.show()



h = (len(dfc1.columns)-1)

#polynomial fit for core 1st order all

q_data = pd.DataFrame() 
qdata = pd.DataFrame()
coefficientsnew7 = pd.DataFrame()
polyx7 = pd.DataFrame()
new_z7 = pd.DataFrame()
for i in range(1, h+1):

    x_data, q_data[i] = (dfcore1['DEG'], dfc1[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew7[i] = np.polyfit(x_data, q_data[i], 120)
    polyx7 = np.poly1d(coefficientsnew7[i])

    new_z7[i] = polyx7(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, q_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z7[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()

#polynomial fit for core 1st order specified range


v_data = pd.DataFrame() 
vdata = pd.DataFrame()
coefficientsnew8 = pd.DataFrame()
polyx8 = pd.DataFrame()
new_z8 = pd.DataFrame()
for i in range(1, h+1):

    x_data, v_data[i] = (dfcore1['DEG'].loc[240:464], dfcr1[i])
    #xdata =x_data/max(x_data)
    #ydata[i] =y_data[i]/max(y_data[i])
    coefficientsnew8[i] = np.polyfit(x_data, v_data[i], 120)
    polyx8 = np.poly1d(coefficientsnew8[i])

    new_z8[i] = polyx8(x_data)

    plt.figure(figsize=(8,5))


plt.figure()
plt.plot(x_data, v_data[:], 'r-', label = 'data')
plt.ylabel('Intensity for Experimental')
plt.xlabel('azimuthal angle')
plt.show()


plt.figure()
plt.plot(x_data, new_z8[:], 'r-', label = 'data')
plt.ylabel('Intensity for Fit')
plt.xlabel('azimuthal angle')
plt.show()


#core root 3 at different angular positions
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('Angles at which peak maxima occur for root 3 sets of peaks order core orientation')
dfcore3maxangle = dfcr3.idxmax()
dfcore32maxangle = dfcr32.idxmax()
x = np.arange(1,(len(dfcore3maxangle) + 1),1)
dfcore3maxanglenew = pd.DataFrame(dfcore3maxangle).to_numpy()
dfcore32maxanglenew = pd.DataFrame(dfcore32maxangle).to_numpy()
l1 = ax1.plot(x, dfcore3maxanglenew, 'r.', label = 'Core root 3 max peak angle 90 degrees')
l2 = ax1.plot(x, dfcore32maxanglenew, 'g.',  label = 'Core root 3 max peak angle 270 degrees')
ax1.legend()
ax1.set_ylabel('Azimuthal angle')
ax1.set_xlabel('Frame number')
plt.show()

#core 1st order at different angular positions
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('Angles at which peak maxima occur for 1st order sets of peaks order core orientation')
dfcore1maxangle = dfcr1.idxmax()
dfcore12maxangle = dfcr12.idxmax()
x = np.arange(1,(len(dfcore1maxangle) + 1),1)
dfcore1maxanglenew = pd.DataFrame(dfcore1maxangle).to_numpy()
dfcore12maxanglenew = pd.DataFrame(dfcore12maxangle).to_numpy()
l1 = ax1.plot(x, dfcore1maxanglenew, 'b.', label = 'Core 1st order max peak angle 90 degrees')
l2 = ax1.plot(x, dfcore12maxanglenew, 'y.',  label = 'Core 1st order max peak angle 270 degrees')
ax1.legend()
ax1.set_ylabel('Azimuthal angle')
ax1.set_xlabel('Frame number')
plt.show()

#1st order and root 3 at 90 degrees
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('Angles at which peak maxima occur for root 3 and 1st order core orientation at 90 degrees')
dfcore3maxangle = dfcr3.idxmax()
dfcore1maxangle = dfcr1.idxmax()
x = np.arange(1,(len(dfcore1maxangle) + 1),1)
dfcore1maxanglenew = pd.DataFrame(dfcore1maxangle).to_numpy()
dfcore3maxanglenew = pd.DataFrame(dfcore3maxangle).to_numpy()
l1 = ax1.plot(x, dfcore3maxanglenew, 'r.', label = 'Core root 3 max peak angle')
l2 = ax1.plot(x, dfcore1maxanglenew, 'b.',  label = 'Core 1st order max peak angle')
ax1.legend()
ax1.set_ylabel('Azimuthal angle')
ax1.set_xlabel('Frame number')
plt.show()

#1st order and root 3 at 270 degrees
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('Angles at which peak maxima occur for root 3 and 1st order core orientation at 270 degrees')
l1 = ax1.plot(x, dfcore32maxanglenew, 'g.', label = 'Core root 3 max peak angle')
l2 = ax1.plot(x, dfcore12maxanglenew, 'y.',  label = 'Core 1st order max peak angle')
ax1.legend()
ax1.set_ylabel('Azimuthal angle')
ax1.set_xlabel('Frame number')
plt.show()


x = np.arange(1,(len(dfcore1maxangle) + 1),1)
print(x)
print(len(dfcore1maxangle))
#integrals

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


areacore1cl = []
areaskin1cl = []
areacoreroot3cl = []
areaskinroot3cl = []
# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
 
    z1 = dfs1.iloc[:,i]
    z2 = dfc1.iloc[:,i]
    z3 = dfsr3clean.iloc[:,i]
    z4 = dfcr3clean.iloc[:,i]

    z5 = dfdegrees1

    skin1trapc=trapz(z1, z5)
    core1trapc=trapz(z2,z5)
    skinroottrapc = trapz(z3, z5)
    coreroottrapc = trapz(z4,z5)

  
    areacore1cl.append(core1trapc)
    areaskin1cl.append(skin1trapc)
    areacoreroot3cl.append(coreroottrapc)
    areaskinroot3cl.append(skinroottrapc)

print("areatrap4421storder_array =", areatrap4421storder_array)
print("areatrap442root3_array =", areatrap442root3_array)

fig4, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array442, areacore1cl, 'r.',  marker=".", label='1st order core')
l2 =  ax1.plot(frame_array442, areaskin1cl, 'b.',  marker=".", label= '1st order skin')
l3 =  ax1.plot(frame_array442, areacoreroot3cl, 'g.',  marker=".", label= 'root 3 core')
l4 =  ax1.plot(frame_array442, areaskinroot3cl, 'k.',  marker=".", label= 'root 3 skin')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 743 sample 7 thickness 0.55-0.73 mm')
plt.show()




ratioscore1cl = np.divide(areacore1cl, areacoreroot3cl)
ratioskincl = np.divide(areaskin1cl, areaskinroot3cl)


fig10, axs = plt.subplots(nrows = 1, ncols = 2, constrained_layout = True, figsize=(24,12))
(ax1, ax2) = axs
fig10.suptitle('FILE 743 sample 7 thickness 0.55-0.73 mm')

ax1.plot(frame_array442, ratioscore1cl, 'b.', color='b', marker='.', label='core ratio 1/root3')
ax1.set_title('Ratio 1storder/root3 for core')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Total Azimuthal Integrated Intensity ratio')
#ax2.legend(loc = 'upper right')
ax1.grid()

ax2.plot(frame_array442, ratioskincl, 'r.',  color='r', marker='.', label='skin/core 1st order')
ax2.set_title('Ratio 1storder/root3 for skin')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Total Azimuthal Integrated Intensity ratio')
#ax2.legend(loc = 'upper right')
ax2.grid()

#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()


#FWHM FOR CORE dfcr1 and dfcr3
dfhalfmaxcore1 = dfcr1.max()/2
x = len(dfhalfmaxcore1)

df_listcore1 = [pd.DataFrame() for x in range(1, x+1)]






#dfcr1 start from i-1 and dfhalfmaxcore start from i for i in dflist look at column i+1
for i in range(1, (x+1)):
    #print(i)
    f = i-1
    df_listcore1[i-1] = dfcr1.index[(dfcr1.iloc[:, i-1] <= 1.08*dfhalfmaxcore1[i]) & (0.92*dfhalfmaxcore1[i] <= dfcr1.iloc[:, i-1])]

#extract min and max angles 
core1maxangless = []
core1minangless = []
for i in range(1, (x+1)):
    f = i-1
    core1minangles = df_listcore1[f].min()
    core1maxangles = df_listcore1[f].max()
    core1minangless.append(core1minangles)
    core1maxangless.append(core1maxangles)
    
core1maxanglesss = np.array(core1maxangless)
core1maxanglesss[core1maxanglesss == core1maxanglesss[54]]=0
core1minanglesss = np.array(core1minangless)
core1minanglesss[core1minanglesss == core1minanglesss[54]]=0
print(core1minanglesss[54])
FWHMcore1 = np.subtract(core1maxanglesss, core1minanglesss)    
plt.figure( figsize = (12, 8))
plt.title('1st order core')
plt.plot(frame_array442, core1minanglesss, 'b.', label = 'min angles')
plt.plot(frame_array442, core1maxanglesss, 'r.', label = 'max angles')
plt.ylabel('FWHM angle')
plt.xlabel('frame')
plt.legend()
plt.show()
  

plt.figure(figsize = (12, 8))
plt.title('1st order core FWHM sample 7 cooling')
plt.plot(frame_array442, FWHMcore1, 'b.', label = 'min angles')
plt.ylabel('FWHM')
plt.xlabel('frame')
#plt.legend()
plt.show()  




dfhalfmaxcore3 = dfcr3.max()/2
l = len(dfhalfmaxcore3)

df_listcore3 = [pd.DataFrame() for x in range(1, l+1)]



for i in range(1, (l+1)):
    #print(i)
    f = i-1
    df_listcore3[i-1] = dfcr3.index[(dfcr3.iloc[:, i-1] <= 1.1*dfhalfmaxcore3[i]) & (0.9*dfhalfmaxcore3[i] <= dfcr3.iloc[:, i-1])]

#extract min and max angles 
core3maxangless = []
core3minangless = []
for i in range(1, (l+1)):
    f = i-1
    core3minangles = df_listcore3[f].min()
    core3maxangles = df_listcore3[f].max()
    core3minangless.append(core3minangles)
    core3maxangless.append(core3maxangles)
    
core3maxanglesss = np.array(core3maxangless)
core3maxanglesss[core3maxanglesss == core3maxanglesss[55]] = 0
core3minanglesss = np.array(core3minangless)
core3minanglesss[core3minanglesss == core3minanglesss[55]] = 0
print(core3minanglesss[55])   
print(core3maxanglesss[2])
FWHMcore3 = np.subtract(core3maxanglesss, core3minanglesss)    
plt.figure(figsize = (12, 8))
plt.title('root 3 core')
plt.plot(frame_array442, core3minanglesss, 'b.', label = 'min angles')
plt.plot(frame_array442, core3maxanglesss, 'r.', label = 'max angles')
plt.ylabel('FWHM angle')
plt.xlabel('frame')
plt.legend()
plt.show()
  

plt.figure(figsize = (12, 8))
plt.title('root 3 core FWHM sample 7 cooling')
plt.plot(frame_array442, FWHMcore3, 'b.', label = 'min angles')
plt.ylabel('FWHM')
plt.xlabel('frame')
#plt.legend()
plt.show()   

dfhalfmaxskin1 = dfsr1.max()/2
r = len(dfhalfmaxskin1)

df_listskin1 = [pd.DataFrame() for x in range(1, r+1)]



for i in range(1, (r+1)):
    #print(i)
    f = i-1
    df_listskin1[i-1] = dfsr1.index[(dfsr1.iloc[:, i-1] <= 1.08*dfhalfmaxskin1[i]) & (0.92*dfhalfmaxskin1[i] <= dfsr1.iloc[:, i-1])]

#extract min and max angles 
skin1maxangless = []
skin1minangless = []
for i in range(1, (r+1)):
    f = i-1
    skin1minangles = df_listskin1[f].min()
    skin1maxangles = df_listskin1[f].max()
    skin1minangless.append(skin1minangles)
    skin1maxangless.append(skin1maxangles)
    
skin1maxanglesss = np.array(skin1maxangless)    
skin1maxanglesss[skin1maxanglesss == skin1maxanglesss[54]]=0
skin1minanglesss = np.array(skin1minangless)    
skin1minanglesss[skin1minanglesss == skin1minanglesss[54]]=0
FWHMskin1 = np.subtract(skin1maxanglesss, skin1minanglesss)    
plt.figure(figsize = (12, 8))
plt.title('1st order skin')
plt.plot(frame_array442, skin1minanglesss, 'b.', label = 'min angles')
plt.plot(frame_array442, skin1maxanglesss, 'r.', label = 'max angles')
plt.ylabel('FWHM angle')
plt.xlabel('frame')
plt.legend()
plt.show()
  

plt.figure(figsize = (12, 8))
plt.title('1st order skin FWHM sample 7 cooling')
plt.plot(frame_array442, FWHMskin1, 'b.', label = 'min angles')
plt.ylabel('FWHM')
plt.xlabel('frame')
#plt.legend()
plt.show() 

dfhalfmaxskin3 = dfsr3.max()/2
s = len(dfhalfmaxskin3)

df_listskin3 = [pd.DataFrame() for x in range(1, s+1)]

for i in range(1, (s+1)):
    #print(i)
    f = i-1
    df_listskin3[i-1] = dfsr3.index[(dfsr3.iloc[:, i-1] <= 1.1*dfhalfmaxskin3[i]) & (0.9*dfhalfmaxskin3[i] <= dfsr3.iloc[:, i-1])]

#extract min and max angles 
skin3maxangless = []
skin3minangless = []
for i in range(1, (s+1)):
    f = i-1
    skin3minangles = df_listskin3[f].min()
    skin3maxangles = df_listskin3[f].max()
    skin3minangless.append(skin3minangles)
    skin3maxangless.append(skin3maxangles)
 
skin3maxanglesss = np.array(skin3maxangless)
skin3maxanglesss[skin3maxanglesss == skin3maxanglesss[55]] = 0
skin3minanglesss = np.array(skin3minangless)
skin3minanglesss[skin3minanglesss == skin3minanglesss[55]] = 0
FWHMskin3 = np.subtract(skin3maxanglesss, skin3minanglesss)  
FWHMskin3 = np.where((FWHMskin3 > 25), nan, FWHMskin3)
plt.figure(figsize = (12, 8))
plt.title('root 3 skin')
plt.plot(frame_array442, skin3minanglesss, 'b.', label = 'min angles')
plt.plot(frame_array442, skin3maxanglesss, 'r.', label = 'max angles')
plt.ylabel('FWHM angle')
plt.xlabel('frame')
plt.legend()
plt.show()
  

plt.figure(figsize = (12, 8))
plt.title('root 3 skin FWHM sample 7 cooling')
plt.plot(frame_array442, FWHMskin3, 'b.', label = 'min angles')
plt.ylabel('FWHM')
plt.xlabel('frame')
#plt.legend()
plt.show() 




#hermans orientation factor
#core 1
dfhcore1 = np.multiply(np.multiply(np.cos(dfcore1['DEG']*pi/180), np.cos(dfcore1['DEG']*pi/180)), np.sin(dfcore1['DEG']*pi/180))
dfhcore12 = np.sin(dfcore1['DEG']*pi/180)

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

dfhcore3 = np.multiply(np.multiply(np.cos(dfcoreroot3['DEG']*pi/180), np.cos(dfcoreroot3['DEG']*pi/180)), np.sin(dfcoreroot3['DEG']*pi/180))
dfhcore32 = np.sin(dfcoreroot3['DEG']*pi/180)

dfhcore3.squeeze(axis = None)
dfhcore3.transpose()

df2core3int1 = dfcr3clean.copy()
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
ax1.set_title('FILE 743 - sample 7 thickness 0.55-0.73 mm')
plt.show()



#hermans orientation factor
#core 1
dfhskin1 = np.multiply(np.multiply(np.cos(dfskin1['DEG']*pi/180), np.cos(dfskin1['DEG']*pi/180)), np.sin(dfskin1['DEG']*pi/180))
dfhskin12 = np.sin(dfcore1['DEG']*pi/180)

dfhskin1.squeeze(axis = None)
dfhskin1.transpose()

df2skin1int1 = dfs1.copy()
funcs = lambda x: np.asarray(x) * np.asarray(dfhskin1)
dfexcesss = df2skin1int1.apply(func)

funcsnew = lambda x: np.asarray(x) * np.asarray(dfhskin12)
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

df2skin3int1 = dfsr3clean.copy()
funcs1 = lambda x: np.asarray(x) * np.asarray(dfhskin3)
dfexcesss2 = df2skin3int1.apply(funcs1)

funcsnew1 = lambda x: np.asarray(x) * np.asarray(dfhskin32)
dfexcesss3 = df2skin3int1.apply(funcsnew1)

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
ax1.set_title('FILE 743 - sample 7 thickness 0.55-0.73 mm')
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
ax1.set_title('FILE 743 - sample 7 thickness 0.55-0.73 mm')
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
ax1.set_title('FILE 743 - sample 7 thickness 0.55-0.73 mm')
plt.show()
