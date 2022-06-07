#FILE 442 DOCS
file505_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 505/FILE5051STORDERPEAK.xlsx'
file505_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 505/FILE505ROOT3PEAK.xlsx'
file505_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 505/FILE5052NDORDER.xlsx'

# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz


# Load spreadsheet FILE 442
x4 = pd.ExcelFile(file505_1storder)
x5 = pd.ExcelFile(file505_root3peak)
x6 = pd.ExcelFile(file505_2ndorder)

# Print the sheet names FILE 442
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
#y = dfintensities.iloc[:,80]
#x = dfdegrees
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
    #FILE 442
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


#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig1, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array505, areatrap5051storder_array, marker="None", label='1st order peaks')
l2 =  ax1.plot(frame_array505, areatrap505root3_array, marker="None", label= 'root 3 order peaks')
l3 =  ax1.plot(frame_array505, areatrap5052ndorder_array, marker="None", label= '2nd order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 505')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 442
df25051storder = df5051storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df25051storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))


#PLOT ROOT 3 AND 2ND ORDER PEAKS FOR FILE 442
plt.plot(frame_array505, areatrap505root3_array, color='r', marker='None', ls='-',label='root 3 peaks')
plt.plot(frame_array505, areatrap5052ndorder_array, color='b', marker='None', ls='-',label='2nd order peaks')
plt.title('FILE 505')
plt.xlabel('Frame')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()
    

