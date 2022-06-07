#462 signifies 1st, 510, 2nd, 503 3rd and 496 perpendicular


# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz


#FILE 462 DOCS
file462_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 14 stuff/FILE577/FILE5771STORDER.xlsx'


#FILE 510 DOCS
file510_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 14 stuff/FILE610/FILE6101STORDER.xlsx'

#FILE 503 DOCS
file503_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/sample 14 stuff/FILE631/FILE6311STORDER.xlsx'



# Load spreadsheet FILE 442
xl = pd.ExcelFile(file462_1storder)


# Print the sheet names FILE 442
print(xl.sheet_names)



df4421storder = xl.parse('Sheet2'); 

#xaxis in degrees [azimuthal angles]
dfdegrees=df4421storder.iloc[:,0]


df442intensities1storder = df4421storder.drop(['DEG'], axis=1)

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

# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = df442intensities1storder.iloc[:,i]

    x = dfdegrees
    area4421storder = trapz(y, x)

    areatrap4421storder_array.append(area4421storder)




#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig1, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array442, areatrap4421storder_array, marker="None", label='1st order peaks')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 577')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 442
df24421storder = df4421storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df24421storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))


    





# Load spreadsheet FILE 505
x4 = pd.ExcelFile(file510_1storder)


# Print the sheet names FILE 505
print(x4.sheet_names)




df5051storder = x4.parse('Sheet2'); 

#xaxis in degrees [azimuthal angles]
dfdegrees=df5051storder.iloc[:,0]


df505intensities1storder = df5051storder.drop(['DEG'], axis=1)

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

# Compute the area using the composite trapezoidal rule.
for i in range(n505):
    #FILE 505
    y = df505intensities1storder.iloc[:,i]

    x = dfdegrees
    area5051storder = trapz(y, x)

    areatrap5051storder_array.append(area5051storder)




#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 505
fig2, ax2 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax2.plot(frame_array505, areatrap5051storder_array, marker="None", label='1st order peaks')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax2.legend()
ax2.set_ylabel('Total azimuthal integrated intensity')
ax2.set_xlabel('Frame')
plt.grid()
ax2.set_title('FILE 610')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 505
df25051storder = df5051storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df25051storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))








# Load spreadsheet FILE 499
x7 = pd.ExcelFile(file503_1storder)



df4991storder = x7.parse('Sheet2'); 

#xaxis in degrees [azimuthal angles]
dfdegrees=df4991storder.iloc[:,0]


df499intensities1storder = df4991storder.drop(['DEG'], axis=1)

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

    x = dfdegrees
    area4991storder = trapz(y, x)

    areatrap4991storder_array.append(area4991storder)




#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 499
fig3, ax3 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax3.plot(frame_array499, areatrap4991storder_array, marker="None", label='1st order peaks')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax3.legend()
ax3.set_ylabel('Total azimuthal integrated intensity')
ax3.set_xlabel('Frame')
plt.grid()
ax3.set_title('FILE 631')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 499
df24991storder = df4991storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df24991storder.plot.line(legend=False, grid = True)

fig2, ax2 = plt.subplots(figsize=(10,6))













#CONVERT INTO MEANINGFUL QUANTITIES

SAMPLETHICKNESS=1.6
thickness505_array=np.array(frame_array505, dtype = float)*(SAMPLETHICKNESS/len(frame_array505))

thickness499_array=np.array(frame_array499, dtype = float)*(SAMPLETHICKNESS/len(frame_array499))

thickness442_array=np.array(frame_array442, dtype = float)*(SAMPLETHICKNESS/len(frame_array442))

#TOTAL AREAS
areatotal442file_array = areatrap4421storder_array

areatotal499file_array = areatrap4991storder_array

areatotal505file_array = areatrap5051storder_array


#PLOT TOTAL INTENSITY SUMS VS THICKNESS FOR ALL FILES/442,499,505.472
plt.plot(thickness505_array, areatotal505file_array, color='r', marker='None', ls='-',label='FILE 610')
plt.plot(thickness499_array, areatotal499file_array, color='b', marker='None', ls='-',label='FILE 631')
plt.plot(thickness442_array, areatotal442file_array, color='g', marker='None', ls='-',label='FILE 577')

plt.title('TOTAL AZIMUTHAL INTEGRATED INTENSITY AS A FUNCTION OF SAMPLE THICKNESS')
plt.xlabel('Thickness')
plt.ylabel('Total Azimuthal Integrated Intensity')
plt.legend(loc = 'upper right')
plt.grid()
#plt.axis([np.pi, 2*np.pi, 0, 22])
plt.show()


#EXTENDED WORK FOR DATA ANALYSIS [PEAK RATIOS AND SKIN AND CORE FRACTION METHODOLOGY]




#PUTTING THEM ALL TOGETHER

fig10, axs = plt.subplots(nrows = 6, ncols = 2, constrained_layout = True, figsize=(20,20))
(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8 ), (ax9, ax10), (ax11, ax12) = axs
fig10.suptitle('SAMPLE 14')



l1 =  ax1.plot(frame_array442, areatrap4421storder_array, marker="None", label='1st order peaks')

ax1.legend()
ax1.set_ylabel('Total azimuthal integrated intensity')
ax1.set_xlabel('Frame')
ax1.set_title('FILE 577')
ax1.grid()


l4 =  ax3.plot(frame_array505, areatrap5051storder_array, marker="None", label='1st order peaks')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax3.legend()
ax3.set_ylabel('Total azimuthal integrated intensity')
ax3.set_xlabel('Frame')
ax3.set_title('FILE 610')
ax3.grid()



l7 =  ax5.plot(frame_array499, areatrap4991storder_array, marker="None", label='1st order peaks')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax5.legend()
ax5.set_ylabel('Total azimuthal integrated intensity')
ax5.set_xlabel('Frame')
ax5.grid()
ax5.set_title('FILE 631')





ax7.plot(thickness505_array, areatotal505file_array, color='r', marker='None', ls='-',label='FILE 610')
ax7.plot(thickness499_array, areatotal499file_array, color='b', marker='None', ls='-',label='FILE 631')
ax7.plot(thickness442_array, areatotal442file_array, color='g', marker='None', ls='-',label='FILE 577')
ax7.set_title('TOTAL AZIMUTHAL INTEGRATED INTENSITY AS A FUNCTION OF SAMPLE THICKNESS')
ax7.set_xlabel('Thickness')
ax7.set_ylabel('Total Azimuthal Integrated Intensity')
ax7.legend(loc = 'upper right')
ax7.grid()



fig10.delaxes(axs[5,1]) #The indexing is zero-based here
plt.show()





#plot all frame intensities for all files
fig11, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4) = axs
fig11.suptitle('SAMPLE 14')
df_list = [df24421storder, df25051storder, df24991storder]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()

