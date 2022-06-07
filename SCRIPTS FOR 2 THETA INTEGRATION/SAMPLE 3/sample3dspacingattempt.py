#442 signifies 1st, 505, 2nd, 499 3rd and 472 perpendicular
#FILE FOR D SPACING

# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz ; from math import *


#FILE 462 DOCS
file462_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 3/FILE4422THETA.xlsx'


#FILE 510 DOCS
file510_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 3/FILE5052THETA.xlsx'

#FILE 503 DOCS
file503_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 3/FILE4992THETA.xlsx'

#
file496_1storder='C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/SCRIPTS FOR 2 THETA INTEGRATION/SAMPLE 3/FILE4722THETA.xlsx'



# Load spreadsheet FILE 442
xl = pd.ExcelFile(file462_1storder)


# Print the sheet names FILE 442
print(xl.sheet_names)



df4421storder = xl.parse('Sheet3'); 

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




df5051storder = x4.parse('Sheet3'); 

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





#PLOT ALL FRAME INTENSITIES FOR 1ST ORDER FILE 505
df25051storder = df5051storder.set_index('DEG')
    # df2root3order = dfroot3.set_index('DEG')
    # df2entire = dfentire.set_index('DEG')
    # df22ndorder = df2ndorder.set_index('DEG')
lines1 = df25051storder.plot.line(legend=False, grid = True)

#fig2, ax2 = plt.subplots(figsize=(10,6))








# Load spreadsheet FILE 499
x7 = pd.ExcelFile(file503_1storder)



df4991storder = x7.parse('Sheet3'); 

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



df4961storder = xl.parse('Sheet3'); 

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






#plot all frame intensities for all files
fig11, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4) = axs
ax1.set_xlim(xmin=100, xmax=275)
ax1.set_ylim(ymin=0, ymax=30000)
ax2.set_xlim(xmin=100, xmax=275)
ax2.set_ylim(ymin=0, ymax=30000)
ax3.set_xlim(xmin=100, xmax=275)
ax3.set_ylim(ymin=0, ymax=30000)
ax4.set_xlim(xmin=100, xmax=275)
ax4.set_ylim(ymin=0, ymax=30000)
fig11.suptitle('SAMPLE 3')
df_list = [df24421storder, df25051storder, df24991storder, df24961storder]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()


#plot all frame intensities for all files
fig12, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True, figsize=(12,12))
(ax1, ax2), (ax3, ax4) = axs
ax1.set_xlim(xmin=175, xmax=275)
ax2.set_xlim(xmin=175, xmax=275)
ax3.set_xlim(xmin=175, xmax=275)
ax4.set_xlim(xmin=175, xmax=275)
fig12.suptitle('SAMPLE 3')
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
ax1.set_xlim(xmin=100, xmax=140)
ax1.set_ylim(ymin=0, ymax=1400)
ax2.set_xlim(xmin=100, xmax=140)
ax2.set_ylim(ymin=0, ymax=1400)
ax3.set_xlim(xmin=100, xmax=140)
ax3.set_ylim(ymin=0, ymax=1400)
ax4.set_xlim(xmin=100, xmax=140)
ax4.set_ylim(ymin=0, ymax=1400)
fig13.suptitle('SAMPLE 3')
df_list = [df24421storder, df25051storder, df24991storder, df24961storder]
# plot counter
n = 0
for r in range(2):
    for c in range(2):
        df_list[n].plot(ax=axs[r,c], legend = False)
        n = n + 1
    
       
plt.show()

