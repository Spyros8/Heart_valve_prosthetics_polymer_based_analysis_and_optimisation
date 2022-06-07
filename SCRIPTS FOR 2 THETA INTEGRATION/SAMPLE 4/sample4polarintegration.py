#442 signifies 1st, 505, 2nd, 499 3rd and 472 perpendicular


# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz ; from math import *


#FILE 462 DOCS
file462_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 4/FILE453THETA.xlsx'


#FILE 510 DOCS
file510_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 4/FILE508THETA.xlsx'

#FILE 503 DOCS
file503_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 4/FILE501THETA.xlsx'

#
file496_1storder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 4/FILE484THETA.xlsx'



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

SAMPLETHICKNESS=1
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

plt.setp(axs, xticks= np.arange(0, 101, 5))


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
fig12.suptitle('SAMPLE 4 1st order peaks')
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
ax1.set_xlim(xmin=0.36, xmax=0.54)
ax1.set_ylim(ymin=0, ymax=800)
ax2.set_xlim(xmin=0.36, xmax=0.54)
ax2.set_ylim(ymin=0, ymax=800)
ax3.set_xlim(xmin=0.36, xmax=0.54)
ax3.set_ylim(ymin=0, ymax=800)
ax4.set_xlim(xmin=0.36, xmax=0.54)
ax4.set_ylim(ymin=0, ymax=1700)
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
fig13.suptitle('SAMPLE 4 root 3 and 2nd order peaks')
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

df442root3max = df24421storder.iloc[24:33,:].max()

lines15 = df442root3max.plot.line(legend=False, grid = True, title='FILE442')

df505root3max = df25051storder.iloc[24:33,:].max()

lines16 = df505root3max.plot.line(legend=False, grid = True, title = 'FILE505')

df499root3max = df24991storder.iloc[24:33,:].max()

lines17 = df499root3max.plot.line(legend=False, grid = True, title = 'FILE 499')

df496root3max = df24961storder.iloc[24:33,:].max()

lines18 = df496root3max.plot.line(legend=False, grid = True, title = 'FILE 472')



fig13, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(14,12))
(ax1, ax2), (ax3, ax4) = axs
fig13.suptitle('SAMPLE 4 ROOT 3 PEAK')
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
        df_list[n].plot(ax=axs[r,c], legend = False) #xticks = np.arange(0, 101, 5))
        n = n + 1
    
       
plt.show()


#1st order intensities method

df4421stordermax = df24421storder.iloc[6:13,:].max()

lines19 = df442root3max.plot.line(legend=False, grid = True, title='FILE442')

df5051stordermax = df25051storder.iloc[6:13,:].max()

lines20 = df5051stordermax.plot.line(legend=False, grid = True, title = 'FILE505')

df4991stordermax = df24991storder.iloc[6:13,:].max()

lines21 = df4991stordermax.plot.line(legend=False, grid = True, title = 'FILE 499')

df4961stordermax = df24961storder.iloc[6:13,:].max()

lines22 = df4961stordermax.plot.line(legend=False, grid = True, title = 'FILE 472')

#plot all frame intensities for all files
fig15, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,8))
(ax1, ax2), (ax3, ax4) = axs

fig15.suptitle('SAMPLE 4 1st order peak max')
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


#CURVE FITTING AND PEAK FINDING

#NON LINEAR REGRESSION

#MODEL PARAMETERS

#from scipy.optimize import least_squares


#def fun(x, t, y):
 #   return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t)  + x[3] * np.exp(-x[4] * t) * np.cos(x[5] * t) - y

#def funn(xe, te):
  #  return xe[0] * np.exp(-xe[1] * te) * np.sin(xe[2] * t)  + xe[3] * np.exp(-xe[4] * te) * np.cos(xe[5] * te)

#B = (len(df4421storder.columns)-1)


#y442= pd.DataFrame()
#res_lsq = pd.DataFrame()
#res_robust = pd.DataFrame()
#x0 = pd.DataFrame()
#x = pd.DataFrame()
#y_lsq = pd.DataFrame()
#y_robust = pd.DataFrame()
#for i in range(1, B+1):
 #   x442 = df4421storder['DEG']
  #  y442[i] = df4421storder[i]
   # x0[i] = np.ones(6)
   # res_lsq[i] = least_squares(fun, x0[i], args=(x442, y442[i]))
    #res_robust[i] = least_squares(fun, x0[i], loss='soft_l1', f_scale=0.1, args=(x442, y442[i]))
   # y_lsq[i] = funn(res_lsq[i].x, df4421storder['DEG'])
    #y_robust[i] = funn(res_robust[i].x, df4421storder['DEG'])


#plt.plot(x442, y442 , 'o', label='data')

#plt.plot(x442, y_lsq, label='lsq')
#plt.plot(x442, y_robust, label='robust lsq')
#plt.xlabel('Intensity')
#plt.ylabel('2theta')
#plt.legend()
#plt.show()


#x442 = df4421storder['DEG']
#y442 = df4421storder[30]
#x0 = np.ones(6)
#res_lsq = least_squares(fun, x0, args=(x442, y442))
#res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(x442, y442[i]))
#res_robust.cost
#res_robust.x

#print(df24421storder[20])
#print(len(df4421storder))
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





#df24421storder[20].shape
#x_data, y_data = (df4421storder['DEG'], df4421storder[20])
#def sigmoid(x, Beta_1, Beta_2):
     #y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    # return y
 
#beta_1 = 0.10
#beta_2 = 1990.0

#xdata =x_data/max(x_data)
#ydata[i] =y_data[i]/max(y_data[i])

#from scipy.optimize import curve_fit
#popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
#print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


#msk = np.random.rand(len(df24421storder[40])) < 0.8
#train_x = xdata[msk]
#test_x = xdata[~msk]
#train_y = ydata[msk]
#test_y = ydata[~msk]

#popt, pcov = curve_fit(sigmoid, train_x, train_y)

#y_hat = sigmoid(test_x, *popt)

#print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
#print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
#from sklearn.metrics import r2_score
#print("R2-score: %.2f" % r2_score(y_hat , test_y) )
#b = len(df4421)
#for i in range(1, b+1):
#coefficients = np.polyfit(xdata, ydata, 400)
#print(coefficients)

#poly[i] = np.poly1d(coefficients[i])

#new_y = poly[i](xdata)

#plt.figure(figsize=(8,5))
#y = sigmoid(xdata, *popt)
#plt.plot(xdata, ydata[i], 'r-', label='data')
#plt.show()

#plt.figure(figsize=(8, 5))
#plt.plot(xdata,y, linewidth=5.0, label='fit')
#plt.plot(test_x, y_hat, 'go', label = 'new fit')
#plt.plot(xdata, new_y, 'y-', label = 'polynomial fit')
#plt.legend(loc='best')
#plt.ylabel('Intensity')
#plt.xlabel('2theta')
#plt.show()
for i in range(1, B+1):
    print(i)
from scipy.signal import find_peaks
f = pd.DataFrame()
peaks = pd.DataFrame()
for i in range(1, B+1):
    f[i]= y_data[i]
    peaks[i], _ = find_peaks(f[i], height = [400, 150000])
    
    
print(y_data[0])
    
    
    

x = df4421storder[20]
y = df4421storder[51]
z = df4421storder[49]
peaks1, _ = find_peaks(x, height = 0 )
peaks2, _ = find_peaks(y, height = [600, 150000])
peaks3, _ = find_peaks(z, height = [600, 150000])
plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.plot(peaks2, y[peaks2], "x" )
plt.plot(peaks1, x[peaks1], "x")
plt.plot(peaks3, z[peaks3], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()