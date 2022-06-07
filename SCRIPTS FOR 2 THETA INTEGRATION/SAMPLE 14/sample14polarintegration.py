#442 signifies 1st, 505, 2nd, 499 3rd and 472 perpendicular


# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz ; from math import *


#FILE 462 DOCS
file462_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 14/FILE596THETA.xlsx' 


#FILE 510 DOCS
file510_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 14/FILE610THETA.xlsx'

#FILE 503 DOCS
file503_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 14/FILE631THETA.xlsx'

#
file496_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 14/FILE577THETA.xlsx'



# Load spreadsheet FILE 442
xl = pd.ExcelFile(file462_1storder)


# Print the sheet names FILE 442
print(xl.sheet_names)



df4421storder = xl.parse('Sheet5'); 

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
l1 =  ax1.plot(frame_array442, areatrap4421storder_array, marker="None", label='-')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total polar integrated intensity')
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

#fig2, ax2 = plt.subplots(figsize=(10,6))


    



# Load spreadsheet FILE 505
x4 = pd.ExcelFile(file510_1storder)


# Print the sheet names FILE 505
print(x4.sheet_names)




df5051storder = x4.parse('Sheet5'); 

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
l1 =  ax2.plot(frame_array505, areatrap5051storder_array, marker="None", label='-')
#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax2.legend()
ax2.set_ylabel('Total polar integrated intensity')
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

#fig2, ax2 = plt.subplots(figsize=(10,6))








# Load spreadsheet FILE 499
x7 = pd.ExcelFile(file503_1storder)



df4991storder = x7.parse('Sheet5'); 

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
l1 =  ax3.plot(frame_array499, areatrap4991storder_array, marker="None", label='-')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax3.legend()
ax3.set_ylabel('Total polar integrated intensity')
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

#fig2, ax2 = plt.subplots(figsize=(10,6))





# Load spreadsheet FILE 442
xl = pd.ExcelFile(file496_1storder)


# Print the sheet names FILE 442
print(xl.sheet_names)



df4961storder = xl.parse('Sheet5'); 

#xaxis in degrees [azimuthal angles]
dfdegrees=df4961storder.iloc[:,0]


df496intensities1storder = df4961storder.drop(['DEG'], axis=1)

# The y values.  A numpy array is used here,
# but a python list could also be used.
#area_array=[]
n=len(df496intensities1storder.columns)
#y = dfintensities.iloc[:,80]
#x = dfdegrees
# Compute the area using the composite trapezoidal rule.
#area = trapz(y, x)
#print("area =", area)
frame_array496=[]
for i in range(1, n+1):
    frame=i
    frame_array496.append(frame)

areatrap4961storder_array=[]

# Compute the area using the composite trapezoidal rule.
for i in range(n):
    #FILE 442
    y = df496intensities1storder.iloc[:,i]

    x = dfdegrees
    area4961storder = trapz(y, x)

    areatrap4961storder_array.append(area4961storder)




#PLOT ALL TOTAL INTEGRATED INTENSITIES FOR FILE 442
fig1, ax1 = plt.subplots(figsize=(10,6))
# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1 =  ax1.plot(frame_array496, areatrap4961storder_array, marker="None", label='-')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax1.legend()
ax1.set_ylabel('Total polar integrated intensity')
ax1.set_xlabel('Frame')
plt.grid()
ax1.set_title('FILE 472')
plt.show()



#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 442
df24961storder = df4961storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df24961storder.plot.line(legend=False, grid = True)

#fig2, ax2 = plt.subplots(figsize=(10,6))












#CONVERT INTO MEANINGFUL QUANTITIES

SAMPLETHICKNESS=1.6
thickness505_array=np.array(frame_array505, dtype = float)*(SAMPLETHICKNESS/len(frame_array505))

thickness499_array=np.array(frame_array499, dtype = float)*(SAMPLETHICKNESS/len(frame_array499))

thickness442_array=np.array(frame_array442, dtype = float)*(SAMPLETHICKNESS/len(frame_array442))

thickness496_array=np.array(frame_array496, dtype = float)*(SAMPLETHICKNESS/len(frame_array496))
#TOTAL AREAS
areatotal442file_array = areatrap4421storder_array

areatotal499file_array = areatrap4991storder_array

areatotal505file_array = areatrap5051storder_array

areatotal496file_array = areatrap4961storder_array




#EXTENDED WORK FOR DATA ANALYSIS [PEAK RATIOS AND SKIN AND CORE FRACTION METHODOLOGY]




#PUTTING THEM ALL TOGETHER

fig10, axs = plt.subplots(nrows = 3, ncols = 2, constrained_layout = True, figsize=(20,20))
(ax1, ax2), (ax3, ax4), (ax5, ax6) = axs
fig10.suptitle('SAMPLE 3')



l1 =  ax1.plot(frame_array442, areatrap4421storder_array, marker="None", label='ALL PEAKS')

ax1.legend()
ax1.set_ylabel('Total polar integrated intensity')
ax1.set_xlabel('Frame')
ax1.set_title('FILE 442')
ax1.grid()


l4 =  ax2.plot(frame_array505, areatrap5051storder_array, marker="None", label='ALL PEAKS')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax2.legend()
ax2.set_ylabel('Total polar integrated intensity')
ax2.set_xlabel('Frame')
ax2.set_title('FILE 505')
ax2.grid()



l7 =  ax3.plot(frame_array499, areatrap4991storder_array, marker="None", label='ALL PEAKS')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax3.legend()
ax3.set_ylabel('Total polar integrated intensity')
ax3.set_xlabel('Frame')
ax3.grid()
ax3.set_title('FILE 499')


plt.setp(axs, xticks= np.arange(0, 191, 10))

l8 =  ax4.plot(frame_array496, areatrap4961storder_array, marker="None", label='ALL PEAKS')

#ax.legend((l1, l2), ('1st order peaks', 'root 3 order peaks'), loc='center', shadow=True)
ax4.legend()
ax4.set_ylabel('Total polar integrated intensity')
ax4.set_xlabel('Frame')
ax4.grid()
ax4.set_title('FILE 472')

ax5.plot(thickness505_array, areatotal505file_array, color='r', marker='None', ls='-',label='FILE 505')
ax5.plot(thickness499_array, areatotal499file_array, color='b', marker='None', ls='-',label='FILE 499')
ax5.plot(thickness442_array, areatotal442file_array, color='g', marker='None', ls='-',label='FILE 442')
ax5.plot(thickness496_array, areatotal496file_array, color='y', marker='None', ls='-',label='FILE 472')
ax5.set_title('TOTAL POLAR INTEGRATED INTENSITY AS A FUNCTION OF SAMPLE THICKNESS')
ax5.set_xlabel('Thickness')
ax5.set_ylabel('Total Azimuthal Integrated Intensity')
ax5.legend(loc = 'upper right')
ax5.grid()



fig10.delaxes(axs[2,1]) #The indexing is zero-based here
plt.show()


#plot all frame intensities for all files
fig11, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4) = axs
ax1.set_xlim(xmin=0.18, xmax=0.6)
ax1.set_ylim(ymin=0, ymax=25000)
ax2.set_xlim(xmin=0.18, xmax=0.6)
ax2.set_ylim(ymin=0, ymax=25000)
ax3.set_xlim(xmin=0.18, xmax=0.6)
ax3.set_ylim(ymin=0, ymax=25000)
ax4.set_xlim(xmin=0.18, xmax=0.6)
ax4.set_ylim(ymin=0, ymax=25000)
fig11.suptitle('SAMPLE 3')
df_list = [df24421storder, df25051storder, df24991storder, df24961storder]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()

df24421storder.index.rename('2 theta', inplace = True)
df25051storder.index.rename('2 theta', inplace = True)
df24991storder.index.rename('2 theta', inplace = True)
df24961storder.index.rename('2 theta', inplace = True)
#plot all frame intensities for all files
fig12, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4) = axs
ax1.set_xlim(xmin=0.18, xmax=0.3)
ax2.set_xlim(xmin=0.18, xmax=0.3)
ax3.set_xlim(xmin=0.18, xmax=0.30)
ax4.set_xlim(xmin=0.18, xmax=0.3)
fig12.suptitle('SAMPLE 14 1st order peaks')
ax1.set_title('12 mm from injection pt')
ax1.set_xlabel('2 theta')
ax1.set_ylabel('intensity')
ax2.set_title('27 mm from injection pt')
ax2.set_xlabel('2 theta')
ax2.set_ylabel('intensity')
ax3.set_title('42 mm from injection pt')
ax3.set_xlabel('2 theta')
ax3.set_ylabel('intensity')
ax4.set_title('perpendicular cut')
ax4.set_xlabel('2 theta')
ax4.set_ylabel('intensity')
df_list = [df24421storder, df25051storder, df24991storder, df24961storder]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()


#plot all frame intensities for all files
fig13, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4) = axs
ax1.set_xlim(xmin=0.34, xmax=0.5)
ax1.set_ylim(ymin=0, ymax=1400)
ax2.set_xlim(xmin=0.34, xmax=0.5)
ax2.set_ylim(ymin=0, ymax=2400)
ax3.set_xlim(xmin=0.34, xmax=0.5)
ax3.set_ylim(ymin=0, ymax=1400)
ax4.set_xlim(xmin=0.34, xmax=0.5)
ax4.set_ylim(ymin=0, ymax=1400)
ax1.set_title('12 mm from injection pt')
ax1.set_xlabel('2 theta')
ax1.set_ylabel('intensity')
ax2.set_title('27 mm from injection pt')
ax2.set_xlabel('2 theta')
ax2.set_ylabel('intensity')
ax3.set_title('42 mm from injection pt')
ax3.set_xlabel('2 theta')
ax3.set_ylabel('intensity')
ax4.set_title('perpendicular cut')
ax4.set_xlabel('2 theta')
ax4.set_ylabel('intensity')
fig13.suptitle('SAMPLE 14 root 3 and 2nd order peaks')
df_list = [df24421storder, df25051storder, df24991storder, df24961storder]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()






#WORK IN PROGRESS
#trying to MATCH D SPACING WITH PEAKS AND FRAMES TO IDENTIFY ROOT 3 PEAK CONTAINING FRAMES
#perform with n (defined as f) is 1 for 1 angstrom wavelength
l = 1.0
f=1
dfdegrees2=dfdegrees*pi/(2*180)
sintheta = np.sin(dfdegrees2)
dspacing1 = 1*l/(2*sintheta)
dspacingroot3 = 1*l*sqrt(3)/(2.0*sintheta)

df_index44211 = pd.merge(df442intensities1storder, dspacing1, right_index=True, left_index=True)
df_index44212 = df_index44211.set_index('DEG')

df_index442root31 = pd.merge(df442intensities1storder, dspacingroot3, right_index=True, left_index=True)
df_index442root32 = df_index442root31.set_index('DEG')


df_index50511 = pd.merge(df505intensities1storder, dspacing1, right_index=True, left_index=True)
df_index50512 = df_index50511.set_index('DEG')

df_index505root31 = pd.merge(df505intensities1storder, dspacingroot3, right_index=True, left_index=True)
df_index505root32 = df_index505root31.set_index('DEG')

df_index49911 = pd.merge(df499intensities1storder, dspacing1, right_index=True, left_index=True)
df_index49912 = df_index49911.set_index('DEG')

df_index499root31 = pd.merge(df499intensities1storder, dspacingroot3, right_index=True, left_index=True)
df_index499root32 = df_index499root31.set_index('DEG')

df_index49611 = pd.merge(df496intensities1storder, dspacing1, right_index=True, left_index=True)
df_index49612 = df_index49611.set_index('DEG')

df_index496root31 = pd.merge(df496intensities1storder, dspacingroot3, right_index=True, left_index=True)
df_index496root32 = df_index496root31.set_index('DEG')

#plot all frame intensities for all files
fig14, axs = plt.subplots(nrows = 4, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8) = axs
fig14.suptitle('SAMPLE 3')
df_list = [df_index44212, df_index442root32, df_index50512, df_index505root32, df_index49912, df_index499root32, df_index49612, df_index496root32]
# plot counter
n = 0
for r in range(4):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()



#ROOT 3 PEAK METHOD FOR FRACTIONS

df442root3max = df24421storder.iloc[26:32,:].max()

lines15 = df442root3max.plot.line(legend=False, grid = True, title='FILE442')

df505root3max = df25051storder.iloc[26:32,:].max()

lines16 = df505root3max.plot.line(legend=False, grid = True, title = 'FILE505')

df499root3max = df24991storder.iloc[26:32,:].max()

lines17 = df499root3max.plot.line(legend=False, grid = True, title = 'FILE 499')

df496root3max = df24961storder.iloc[26:32,:].max()

lines18 = df496root3max.plot.line(legend=False, grid = True, title = 'FILE 472')

#plot all frame intensities for all files
fig14, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,8))
(ax1, ax2), (ax3, ax4) = axs

fig14.suptitle('SAMPLE 14 ROOT 3 PEAK')
ax1.set_title('12 mm from injection pt')
ax1.set_xlabel('frame')
ax1.set_ylabel('Maximum root 3 intensity')
ax2.set_title('27 mm from injection pt')
ax2.set_xlabel('frame')
ax2.set_ylabel('Maximum root 3 intensity')
ax3.set_title('42 mm from injection pt')
ax3.set_xlabel('frame')
ax3.set_ylabel('Maximum root 3 intensity')
ax4.set_title('perpendicular cut')
ax4.set_xlabel('frame')
ax4.set_ylabel('Maximum root 3 intensity')
df_list = [df442root3max, df505root3max, df499root3max, df496root3max]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)#, xticks = range(0, 191, 10))
        n = n + 1
    
       
plt.show()


#1st order intensities method

df4421stordermax = df24421storder.iloc[6:12,:].max()

lines19 = df442root3max.plot.line(legend=False, grid = True, title='FILE442')

df5051stordermax = df25051storder.iloc[6:12,:].max()

lines20 = df5051stordermax.plot.line(legend=False, grid = True, title = 'FILE505')

df4991stordermax = df24991storder.iloc[6:12,:].max()

lines21 = df4991stordermax.plot.line(legend=False, grid = True, title = 'FILE 499')

df4961stordermax = df24961storder.iloc[6:12,:].max()

lines22 = df4961stordermax.plot.line(legend=False, grid = True, title = 'FILE 472')

#plot all frame intensities for all files
fig15, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,8))
(ax1, ax2), (ax3, ax4) = axs

fig15.suptitle('SAMPLE 14 1st order peak max')
ax1.set_title('12 mm from injection pt')
ax1.set_xlabel('frame')
ax1.set_ylabel('Maximum 1st order intensity')
ax2.set_title('27 mm from injection pt')
ax2.set_xlabel('frame')
ax2.set_ylabel('Maximum 1st order intensity')
ax3.set_title('42 mm from injection pt')
ax3.set_xlabel('frame')
ax3.set_ylabel('Maximum 1st order intensity')
ax4.set_title('perpendicular cut')
ax4.set_xlabel('frame')
ax4.set_ylabel('Maximum 1st order intensity')
df_list = [df4421stordermax, df5051stordermax, df4991stordermax, df4961stordermax]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()
