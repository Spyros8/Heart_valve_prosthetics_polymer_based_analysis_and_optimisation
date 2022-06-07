#FILE 442 DOCS
file472_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE 472/FILE4721STORDERPEAKS.xlsx'
#file442_root3peak='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/azimuthalroot3peak.xlsx'
#file442_2ndorder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/FILE442/azimuthalsecond order.xlsx'
# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz

# Load spreadsheet FILE 442
xl = pd.ExcelFile(file472_1storder)
#x2 = pd.ExcelFile(file442_root3peak)
#x3 = pd.ExcelFile(file442_2ndorder)

# Print the sheet names FILE 442
#print(xl.sheet_names)
#print(x2.sheet_names)
#print(x3.sheet_names)



df4721storder = xl.parse('Sheet2')

#xaxis in degrees [azimuthal angles]
dfdegrees=df4721storder.iloc[:,0]


df472intensities1storder = df4721storder.drop(['DEG'], axis=1)

# The y values.  A numpy array is used here,
# but a python list could also be used.
#area_array=[]
n=len(df472intensities1storder.columns)
#y = dfintensities.iloc[:,80]
#x = dfdegrees
# Compute the area using the composite trapezoidal rule.
#area = trapz(y, x)
#print("area =", area)
frame_array472=[]
for i in range(1, n+1):
    frame=i
    frame_array472.append(frame)

areatrap4721storder_array=[]

# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = df472intensities1storder.iloc[:,i]
  
    x = dfdegrees
    area4721storder = trapz(y, x)

    areatrap4721storder_array.append(area4721storder)

print("areatrap4721storder_array =", areatrap4721storder_array)



#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig1, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array472, areatrap4721storder_array, marker="None", label='1st order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 472')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 442
df24721storder = df4721storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df24721storder.plot.line(legend=False, grid = True)



