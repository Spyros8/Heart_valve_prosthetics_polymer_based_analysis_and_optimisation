#462 signifies 1st, 510, 2nd, 503 3rd and 496 perpendicular


# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz


#FILE 462 DOCS
file462_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE560/FILE5601STORDER.xlsx'
file462_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE560/FILE560ROOT3PEAK.xlsx'
file462_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE560/FILE5602NDORDER.xlsx'

#FILE 510 DOCS
file510_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE602/FILE6021STORDER.xlsx'
file510_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE602/FILE602ROOT3PEAK.xlsx'
file510_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE602/FILE6022NDORDER.xlsx'

#FILE 503 DOCS
file503_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE614/FILE6141STORDER.xlsx'
file503_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE614/FILE614ROOT3PEAK.xlsx'
file503_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE614/FILE6142NDORDER.xlsx'


#FILE 496 PERPENDICULAR DOCS'C:/U
file496_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE584/FILE5841STORDER.xlsx'
file496_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE584/FILE584ROOT3PEAK.xlsx'
file496_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 11 stuff/FILE584/FILE5842NDORDER.xlsx'






# Load spreadsheet FILE 442
xl = pd.ExcelFile(file462_1storder)
x2 = pd.ExcelFile(file462_root3peak)
x3 = pd.ExcelFile(file462_2ndorder)

# Print the sheet names FILE 442
print(xl.sheet_names)
print(x2.sheet_names)
print(x3.sheet_names)



df4421storder = xl.parse('Sheet5'); df442root3 = x2.parse('Sheet5'); df4422ndorder = x3.parse('Sheet5')

#xaxis in degrees [azimuthal angles]
dfdegrees=df4421storder.iloc[:,0]


df442intensities1storder = df4421storder.drop(['DEG'], axis=1)
df442intensitiesroot3 = df442root3.drop(['DEG'], axis=1)
df442intensities2ndorder = df4422ndorder.drop(['DEG'], axis=1)
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
areatrap4422ndorder_array=[]
# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = df442intensities1storder.iloc[:,i]
    z = df442intensitiesroot3.iloc[:,i]
    u= df442intensities2ndorder.iloc[:,i]
    x = dfdegrees
    area4421storder = trapz(y, x)
    area442root3 = trapz(z,x)
    area4422ndorder = trapz(u,x)
    areatrap4421storder_array.append(area4421storder)
    areatrap442root3_array.append(area442root3)
    areatrap4422ndorder_array.append(area4422ndorder)



#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig1, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array442, areatrap4421storder_array, marker="None", label='1st order peaks')
l2 =  ax1.plot(frame_array442, areatrap442root3_array, marker="None", label= 'root 3 order peaks')
l3 =  ax1.plot(frame_array442, areatrap4422ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 560')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 442
df24421storder = df4421storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df24421storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))


#PLOT ROOT 3 AND 2ND ORDER PEAKS FOR FILE 442
plt.plot(frame_array442, areatrap442root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
plt.plot(frame_array442, areatrap4422ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
plt.title('FILE 560')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()
    





# Load spreadsheet FILE 505
x4 = pd.ExcelFile(file510_1storder)
x5 = pd.ExcelFile(file510_root3peak)
x6 = pd.ExcelFile(file510_2ndorder)

# Print the sheet names FILE 505
print(x4.sheet_names)
print(x5.sheet_names)
print(x6.sheet_names)



df5051storder = x4.parse('Sheet5'); df505root3 = x5.parse('Sheet5'); df5052ndorder = x6.parse('Sheet5')

#xaxis in degrees [azimuthal angles]
dfdegrees=df5051storder.iloc[:,0]


df505intensities1storder = df5051storder.drop(['DEG'], axis=1)
df505intensitiesroot3 = df505root3.drop(['DEG'], axis=1)
df505intensities2ndorder = df5052ndorder.drop(['DEG'], axis=1)
# The y values.  A numpy array is used here,
# but a python list could also be used.
#area_array=[]
n505=len(df505intensities1storder.columns)

# Compute the area using the composite trapezoidal rule.
#area = trapz(y, x)
#print("area =", area)
frame_array505=[]
for i in range(1, n505+1):
    frame=i
    frame_array505.append(frame)

areatrap5051storder_array=[]
areatrap505root3_array=[]
areatrap5052ndorder_array=[]
# Compute the area using the composite trapezoidal rule.
for i in range(n505):
    #FILE 505
    y = df505intensities1storder.iloc[:,i]
    z = df505intensitiesroot3.iloc[:,i]
    u= df505intensities2ndorder.iloc[:,i]
    x = dfdegrees
    area5051storder = trapz(y, x)
    area505root3 = trapz(z,x)
    area5052ndorder = trapz(u,x)
    areatrap5051storder_array.append(area5051storder)
    areatrap505root3_array.append(area505root3)
    areatrap5052ndorder_array.append(area5052ndorder)



#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 505
fig2, ax2 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax2.plot(frame_array505, areatrap5051storder_array, marker="None", label='1st order peaks')
l2 =  ax2.plot(frame_array505, areatrap505root3_array, marker="None", label= 'root 3 order peaks')
l3 =  ax2.plot(frame_array505, areatrap5052ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax2.legend()
ax2.set_ylabel('Total azimuthal integrated intensity')
ax2.set_xlabel('Frame')
plt.grid()
ax2.set_title('FILE 602')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 505
df25051storder = df5051storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df25051storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))


#PLOT ROOT 3 AND 2ND ORDER PEAKS FOR FILE 505
plt.plot(frame_array505, areatrap505root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
plt.plot(frame_array505, areatrap5052ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
plt.title('FILE 602')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()






# Load spreadsheet FILE 499
x7 = pd.ExcelFile(file503_1storder)
x8 = pd.ExcelFile(file503_root3peak)
x9 = pd.ExcelFile(file503_2ndorder)


df4991storder = x7.parse('Sheet5'); df499root3 = x8.parse('Sheet5'); df4992ndorder = x9.parse('Sheet5')

#xaxis in degrees [azimuthal angles]
dfdegrees=df4991storder.iloc[:,0]


df499intensities1storder = df4991storder.drop(['DEG'], axis=1)
df499intensitiesroot3 = df499root3.drop(['DEG'], axis=1)
df499intensities2ndorder = df4992ndorder.drop(['DEG'], axis=1)
# The y values.  A numpy array is used here,
# but a python list could also be used.
#area_array=[]
n499=len(df499intensities1storder.columns)
#y = dfintensities.iloc[:,80]
#x = dfdegrees
# Compute the area using the composite trapezoidal rule.
#area = trapz(y, x)
#print("area =", area)
frame_array499=[]
for i in range(1, n499+1):
    frame=i
    frame_array499.append(frame)

areatrap4991storder_array=[]
areatrap499root3_array=[]
areatrap4992ndorder_array=[]
# Compute the area using the composite trapezoidal rule.
for i in range(n499):
    #FILE 442
    y = df499intensities1storder.iloc[:,i]
    z = df499intensitiesroot3.iloc[:,i]
    u= df499intensities2ndorder.iloc[:,i]
    x = dfdegrees
    area4991storder = trapz(y, x)
    area499root3 = trapz(z,x)
    area4992ndorder = trapz(u,x)
    areatrap4991storder_array.append(area4991storder)
    areatrap499root3_array.append(area499root3)
    areatrap4992ndorder_array.append(area4992ndorder)



#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 499
fig3, ax3 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax3.plot(frame_array499, areatrap4991storder_array, marker="None", label='1st order peaks')
l2 =  ax3.plot(frame_array499, areatrap499root3_array, marker="None", label= 'root 3 order peaks')
l3 =  ax3.plot(frame_array499, areatrap4992ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax3.legend()
ax3.set_ylabel('Total azimuthal integrated intensity')
ax3.set_xlabel('Frame')
plt.grid()
ax3.set_title('FILE 614')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 499
df24991storder = df4991storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df24991storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))


#PLOT ROOT 3 AND 2ND ORDER PEAKS FOR FILE 499
plt.plot(frame_array499, areatrap499root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
plt.plot(frame_array499, areatrap4992ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
plt.title('FILE 614')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()







# Load spreadsheet FILE 472
x10 = pd.ExcelFile(file496_1storder)
x11 = pd.ExcelFile(file496_root3peak)
x12 = pd.ExcelFile(file496_2ndorder)


df4721storder = x10.parse('Sheet5'); df472root3 = x11.parse('Sheet5'); df4722ndorder = x12.parse('Sheet5')

#xaxis in degrees [azimuthal angles]
dfdegrees=df4721storder.iloc[:,0]


df472intensities1storder = df4721storder.drop(['DEG'], axis=1)
df472intensitiesroot3 = df472root3.drop(['DEG'], axis=1)
df472intensities2ndorder = df4722ndorder.drop(['DEG'], axis=1)
# The y values.  A numpy array is used here,
# but a python list could also be used.
#area_array=[]
n472=len(df472intensities1storder.columns)
#y = dfintensities.iloc[:,80]
#x = dfdegrees
# Compute the area using the composite trapezoidal rule.
#area = trapz(y, x)
#print("area =", area)
frame_array472=[]
for i in range(1, n472+1):
    frame=i
    frame_array472.append(frame)

areatrap4721storder_array=[]
areatrap472root3_array=[]
areatrap4722ndorder_array=[]
# Compute the area using the composite trapezoidal rule.
for i in range(n472):
    #FILE 442
    y = df472intensities1storder.iloc[:,i]
    z = df472intensitiesroot3.iloc[:,i]
    u= df472intensities2ndorder.iloc[:,i]
    x = dfdegrees
    area4721storder = trapz(y, x)
    area472root3 = trapz(z,x)
    area4722ndorder = trapz(u,x)
    areatrap4721storder_array.append(area4721storder)
    areatrap472root3_array.append(area472root3)
    areatrap4722ndorder_array.append(area4722ndorder)



#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 472
fig4, ax4 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax4.plot(frame_array472, areatrap4721storder_array, marker="None", label='1st order peaks')
l2 =  ax4.plot(frame_array472, areatrap472root3_array, marker="None", label= 'root 3 order peaks')
l3 =  ax4.plot(frame_array472, areatrap4722ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax4.legend()
ax4.set_ylabel('Total azimuthal integrated intensity')
ax4.set_xlabel('Frame')
plt.grid()
ax4.set_title('FILE 584')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 472
df24721storder = df4721storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df24721storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))


#PLOT ROOT 3 AND 2ND ORDER PEAKS FOR FILE 472
plt.plot(frame_array472, areatrap472root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
plt.plot(frame_array472, areatrap4722ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
plt.title('FILE 584')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()










#CONVERT INTO MEANINGFUL QUANTITIES

SAMPLETHICKNESS=1.6
thickness505_array=np.array(frame_array505, dtype = float)*(SAMPLETHICKNESS/len(frame_array505))

thickness499_array=np.array(frame_array499, dtype = float)*(SAMPLETHICKNESS/len(frame_array499))

thickness442_array=np.array(frame_array442, dtype = float)*(SAMPLETHICKNESS/len(frame_array442))
thickness472_array=np.array(frame_array472, dtype = float)*(SAMPLETHICKNESS/len(frame_array472))
#TOTAL AREAS
areatotal442file_array = np.add(np.add(areatrap4421storder_array, areatrap442root3_array), areatrap4422ndorder_array)

areatotal499file_array = np.add(np.add(areatrap4991storder_array, areatrap499root3_array), areatrap4992ndorder_array)

areatotal505file_array = np.add(np.add(areatrap5051storder_array, areatrap505root3_array), areatrap5052ndorder_array)

areatotal472file_array = np.add(np.add(areatrap4721storder_array, areatrap472root3_array), areatrap4722ndorder_array)
#PLOT TOTAL INTENSITY SUMS VS THICKNESS FOR ALL FILES/442,499,505.472
plt.plot(thickness505_array, areatotal505file_array, color='r', marker='None', ls='-',label='FILE 602')
plt.plot(thickness499_array, areatotal499file_array, color='b', marker='None', ls='-',label='FILE 614')
plt.plot(thickness442_array, areatotal442file_array, color='g', marker='None', ls='-',label='FILE 560')
plt.plot(thickness472_array, areatotal472file_array, color='y', marker='None', ls='-',label='FILE 584')
plt.title('TOTAL AZIMUTHAL INTEGRATED INTENSITY AS A FUNCTION OF SAMPLE THICKNESS')
plt.xlabel('Thickness')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()


#EXTENDED WORK FOR DATA ANALYSIS [PEAK RATIOS AND SKIN AND CORE FRACTION METHODOLOGY]

#DIVIDE NUMPY ARRAYS 1ST ORDER AND ROOT 3 IN SAME ORDER AS THEY APPEAR AT THE TOP OF THE FILE

file442ratio1 = np.divide(areatrap4421storder_array, areatrap442root3_array)
file505ratio1 = np.divide(areatrap5051storder_array, areatrap505root3_array)
file499ratio1 = np.divide(areatrap4991storder_array, areatrap499root3_array)
file472ratio1 = np.divide(areatrap4721storder_array, areatrap472root3_array)

#DIVIDE NUMPY ARRAYS ROOT 3 AND 2ND ORDER IN SAME ORDER AS THEY APPEAR AT THE TOP OF THE FILE

file442ratio2 = np.divide(areatrap442root3_array, areatrap4422ndorder_array)
file505ratio2 = np.divide(areatrap505root3_array, areatrap5052ndorder_array)
file499ratio2 = np.divide(areatrap499root3_array, areatrap4992ndorder_array)
file472ratio2 = np.divide(areatrap472root3_array, areatrap4722ndorder_array)

#PUTTING THEM ALL TOGETHER

fig10, axs = plt.subplots(nrows = 5, ncols = 2, constrained_layout = True, figsize=(20,20))
(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8 ), (ax9, ax10) = axs
fig10.suptitle('Sample 11')



l1 =  ax1.plot(frame_array442, areatrap4421storder_array, marker="None", label='1st order peaks')
l2 =  ax1.plot(frame_array442, areatrap442root3_array,  marker="None", label= 'root 3 order peaks')
l3 =  ax1.plot(frame_array442, areatrap4422ndorder_array, marker="None", label= '2nd order peaks')
ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
ax1.set_title('1.47 mm thickness, perpendicular cut of sample')
#ax1.grid()

ax2.plot(frame_array442, areatrap442root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
ax2.plot(frame_array442, areatrap4422ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
ax2.set_title('1.47 mm thickness, perpendicular cut of sample')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Total Azimuthal Integrated Intensity')
ax2.legend(loc = 'upper right')
#ax2.grid()

l4 =  ax3.plot(frame_array505, areatrap5051storder_array, marker="None", label='1st order peaks')
l5 =  ax3.plot(frame_array505, areatrap505root3_array, marker="None", label= 'root 3 order peaks')
l6 =  ax3.plot(frame_array505, areatrap5052ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax3.legend()
ax3.set_ylabel('Total azimuthal integrated intensity')
ax3.set_xlabel('Frame')
ax3.set_title('1.51 mm thickness, 27 mm from injection point')
#ax3.grid()

ax4.plot(frame_array505, areatrap505root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
ax4.plot(frame_array505, areatrap5052ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
ax4.set_title('1.51 mm thickness, 27 mm from injection point')
ax4.set_xlabel('Frame')
ax4.set_ylabel('Total Azimuthal Integrated Intensity')
ax4.legend(loc = 'upper right')
#ax4.grid()

l7 =  ax5.plot(frame_array499, areatrap4991storder_array, marker="None", label='1st order peaks')
l8 =  ax5.plot(frame_array499, areatrap499root3_array, marker="None", label= 'root 3 order peaks')
l9 =  ax5.plot(frame_array499, areatrap4992ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax5.legend()
ax5.set_ylabel('Total azimuthal integrated intensity')
ax5.set_xlabel('Frame')
#ax5.grid()
ax5.set_title('1.46 mm thickness, 42 mm from injection point')

ax6.plot(frame_array499, areatrap499root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
ax6.plot(frame_array499, areatrap4992ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
ax6.set_title('1.46 mm thickness, 42 mm from injection point')
ax6.set_xlabel('Frame')
ax6.set_ylabel('Total Azimuthal Integrated Intensity')
ax6.legend(loc = 'upper right')
#ax6.grid()

l10 =  ax7.plot(frame_array472, areatrap4721storder_array, marker="None", label='1st order peaks')
l11 =  ax7.plot(frame_array472, areatrap472root3_array, marker="None", label= 'root 3 order peaks')
l12 =  ax7.plot(frame_array472, areatrap4722ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax7.legend()
ax7.set_ylabel('Total azimuthal integrated intensity')
ax7.set_xlabel('Frame')
ax7.legend(loc = 'upper right')
ax7.set_title('1.47 mm thickness, 12 mm from injection point')
#ax7.grid()

ax8.plot(frame_array472, areatrap472root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
ax8.plot(frame_array472, areatrap4722ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
ax8.set_title('1.47 mm thickness, 12 mm from injection point')
ax8.set_xlabel('Frame')
ax8.set_ylabel('Total Azimuthal Integrated Intensity')
ax8.legend(loc = 'upper right')
#ax8.grid()



ax9.plot(frame_array442, file442ratio1, 'g.', color='g', marker='.',label='12 mm from injection')
ax9.plot(frame_array505, file505ratio1, 'r.',  color='r', marker='.',label='27 mm from injection')
ax9.plot(frame_array499, file499ratio1, 'b.', color='b', marker='.',label='42 mm from injection')
ax9.plot(frame_array472, file472ratio1, 'k.', color='k', marker='.',label='perpendicular cut sample')
ax9.set_title('RATIO OF 1ST ORDER TO ROOT 3 INTENSITIES')
ax9.set_xlabel('Frame')
ax9.set_ylabel('Intensity ratio')
ax9.legend(loc = 'upper right')
#ax9.grid()

ax10.plot(frame_array442, file442ratio2, 'g.',  color='g', marker='.',label='12 mm from injection')
ax10.plot(frame_array505, file505ratio2, 'r.', color='r', marker='.',label='27 mm from injection')
ax10.plot(frame_array499, file499ratio2, 'b.', color='b', marker='.',label='42 mm from injection')
ax10.plot(frame_array472, file472ratio2, 'k.', color='k', marker='.',label='perpendicular cut sample')
ax10.set_title('RATIO OF ROOT 3 TO 2ND ORDER INTENSITIES')
ax10.set_xlabel('Frame')
ax10.set_ylabel('Intensity ratio')
ax10.legend(loc = 'upper right')
#ax10.grid()

plt.show()





#plot all frame intensities for all files
fig11, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4) = axs
fig11.suptitle('SAMPLE 11')
df_list = [df24421storder, df25051storder, df24991storder, df24721storder]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()

