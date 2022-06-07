# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz


#FILE 442 DOCS
file442_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/INTENSITYVSTWOTHETA1storder-Im442.xlsx'
file442_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/azimuthalroot3peak.xlsx'
file442_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/azimuthalsecond order.xlsx'

#FILE 505 DOCS
file505_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 505/FILE5051STORDERPEAK.xlsx'
file505_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 505/FILE505ROOT3PEAK.xlsx'
file505_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 505/FILE5052NDORDER.xlsx'

#FILE 499 DOCS
file499_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 499/FILE4991STORDERPEAKS.xlsx'
file499_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 499/FILE499ROOT3PEAKS.xlsx'
file499_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 499/FILE4992NDORDERPEAKS.xlsx'



# Load spreadsheet FILE 442
xl = pd.ExcelFile(file442_1storder)
x2 = pd.ExcelFile(file442_root3peak)
x3 = pd.ExcelFile(file442_2ndorder)

# Print the sheet names FILE 442
print(xl.sheet_names)
print(x2.sheet_names)
print(x3.sheet_names)



df4421storder = xl.parse('Sheet1'); df442root3 = x2.parse('Sheet1'); df4422ndorder = x3.parse('Sheet1')

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
print("areatrap4421storder_array =", areatrap4421storder_array)
print("areatrap442root3_array =", areatrap442root3_array)
print("areatrap4422ndorder_array =", areatrap4422ndorder_array)


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
ax1.set_title('FILE 442')
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
plt.title('FILE 442')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()
    











# Load spreadsheet FILE 505
x4 = pd.ExcelFile(file505_1storder)
x5 = pd.ExcelFile(file505_root3peak)
x6 = pd.ExcelFile(file505_2ndorder)

# Print the sheet names FILE 505
print(x4.sheet_names)
print(x5.sheet_names)
print(x6.sheet_names)



df5051storder = x4.parse('Sheet1'); df505root3 = x5.parse('Sheet1'); df5052ndorder = x6.parse('Sheet1')

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
print("areatrap5051storder_array =", areatrap5051storder_array)
print("areatrap505root3_array =", areatrap505root3_array)
print("areatrap5052ndorder_array =", areatrap5052ndorder_array)


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
ax2.set_title('FILE 505')
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
plt.title('FILE 505')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()

















# Load spreadsheet FILE 499
x7 = pd.ExcelFile(file499_1storder)
x8 = pd.ExcelFile(file499_root3peak)
x9 = pd.ExcelFile(file499_2ndorder)


df4991storder = x7.parse('Sheet1'); df499root3 = x8.parse('Sheet1'); df4992ndorder = x9.parse('Sheet1')

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
print("areatrap4991storder_array =", areatrap4991storder_array)
print("areatrap499root3_array =", areatrap499root3_array)
print("areatrap4992ndorder_array =", areatrap4992ndorder_array)


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
ax3.set_title('FILE 499')
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
plt.title('FILE 499')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()





#CONVERT INTO MEANINGFUL QUANTITIES

SAMPLETHICKNESS=1.0
thickness505_array=np.array(frame_array505, dtype = float)*(SAMPLETHICKNESS/len(frame_array505))

thickness499and442_array=np.array(frame_array499, dtype = float)*(SAMPLETHICKNESS/len(frame_array499))

#TOTAL AREAS
areatotal442file_array = np.add(np.add(areatrap4421storder_array, areatrap442root3_array), areatrap4422ndorder_array)

areatotal499file_array = np.add(np.add(areatrap4991storder_array, areatrap499root3_array), areatrap4992ndorder_array)

areatotal505file_array = np.add(np.add(areatrap5051storder_array, areatrap505root3_array), areatrap5052ndorder_array)



#PLOT TOTAL INTENSITY SUMS VS THICKNESS FOR ALL FILES/442,499,505
plt.plot(thickness505_array, areatotal505file_array, color='r', marker='None', ls='-',label='FILE 505')
plt.plot(thickness499and442_array, areatotal499file_array, color='b', marker='None', ls='-',label='FILE 499')
plt.plot(thickness499and442_array, areatotal442file_array, color='b', marker='None', ls='-',label='FILE 442')
plt.title('TOTAL AZIMUTHAL INTEGRATED INTENSITY AS A FUNCTION OF SAMPLE THICKNESS')
plt.xlabel('Thickness')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()