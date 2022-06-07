#FILE 442 DOCS
file442_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/INTENSITYVSTWOTHETA1storder-Im442.xlsx'
file442_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/azimuthalroot3peak.xlsx'
file442_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/azimuthalsecond order.xlsx'
# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz

# Load spreadsheet FILE 442
xl = pd.ExcelFile(file442_1storder)
x2 = pd.ExcelFile(file442_root3peak)
x3 = pd.ExcelFile(file442_2ndorder)

# Print the sheet names FILE 442
print(xl.sheet_names)
print(x2.sheet_names)
print(x3.sheet_names)



df4421storder = xl.parse('Sheet2'); df442root3 = x2.parse('Sheet3'); df4422ndorder = x3.parse('Sheet3')

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