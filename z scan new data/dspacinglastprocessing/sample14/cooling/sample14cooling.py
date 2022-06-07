# Import pandas
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz
from math import *

#FILE 442 DOCS
file442_1storder= 'C:/Users/Spyros/.spyder-py3/RESEARCH PROJECT RELEVANT IIB/z scan new data/dspacinglastprocessing/sample14/cooling/832835dsp.xlsx'


# Load spreadsheet FILE 442
x1 = pd.ExcelFile(file442_1storder)

# Print the sheet names FILE 442
print(x1.sheet_names)




dfc1270 = x1.parse('core1270'); dfc190 = x1.parse('core190'); dfc3270 = x1.parse('coreroot3270'); dfc390 = x1.parse('coreroot390')

dfs10 = x1.parse('skin10'); dfs1180 = x1.parse('skin1180'); dfs30 = x1.parse('skinroot30'); dfs3180 = x1.parse('skinroot3180')
#xaxis in degrees [azimuthal angles]



df2c1270=dfc1270.set_index('DEG')
df2c190=dfc190.set_index('DEG')
df2c3270=dfc3270.set_index('DEG')
df2c390=dfc390.set_index('DEG')



df2s1180=dfs1180.set_index('DEG')
df2s10=dfs10.set_index('DEG')
df2s3180=dfs3180.set_index('DEG')
df2s30=dfs30.set_index('DEG')



(df2c1270.iloc[59:95,:]-40).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'core1 270 ' )

(df2c190.iloc[60:95,:]-40).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'core1 90 ' )

(df2c3270.iloc[105:145,:]-40).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'coreroot3 270 ' )

(df2c390.iloc[105:145,:]-40).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'coreroot3 90' )


(df2s10.iloc[60:90,:]-40).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'skin1 0 ' )

(df2s1180.iloc[60:90,:]-40).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'skin1 180 ' )

(df2s3180.iloc[115:137,:]-0).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'skinroot3 180 ' )

(df2s30.iloc[115:137,:]- 0).plot(figsize=[12,12], legend= False, ylim = [0, 1000], title = 'skinroot3 0 ' )

core190 = (df2c190.iloc[60:95,:]-40)
core190[core190 < 0]=0
core1270 = (df2c1270.iloc[59:95,:]-40)
core1270[core1270 < 0]=0
core3270 = (df2c3270.iloc[105:145,:]-40)
core3270[core3270 < 0] = 0
core390 = (df2c390.iloc[105:145,:] - 40)
core390[core390 < 0] = 0

skin10 = (df2s10.iloc[60:90,:]-40)
skin10[skin10 < 0] = 0
skin1180 = (df2s1180.iloc[60:90,:]-40)
skin1180[skin1180 < 0]=0
skin3180 = (df2s3180.iloc[115:137,:]-0)
skin3180[skin3180<0]=0
skin30 = (df2s30.iloc[115:137,:]-0)
skin30[skin30<0]=0


dfcore190dsp = core190.idxmax()
dfcore1270dsp = core1270.idxmax()
dfcore390dsp = core390.idxmax()
dfcore3270dsp = core3270.idxmax()

dfskin10dsp = skin10.idxmax()
dfskin1180dsp = skin1180.idxmax()
dfskin30dsp = skin30.idxmax()
dfskin3180dsp = skin3180.idxmax()

#core 1st order at different angular positions
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('D spacing per frame for core 1st order sample 7 cooling')
core190dsp = pd.DataFrame(dfcore190dsp).to_numpy()
core1270dsp = pd.DataFrame(dfcore1270dsp).to_numpy()
x = np.arange(1,(len(dfcore190dsp) + 1),1)
l1 = ax1.plot(x, core190dsp, 'b-', label = 'Core 1st order at 90 degrees')
l2 = ax1.plot(x, core1270dsp, 'r-', label = 'Core 1st order at 270 degrees')
ax1.legend()
ax1.set_ylabel('D spacing')
ax1.set_xlabel('Frame number')
plt.show()



#core 1st order at different angular positions
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('D spacing per frame for core root 3 sample 7 cooling')
core390dsp = pd.DataFrame(dfcore390dsp).to_numpy()
core3270dsp = pd.DataFrame(dfcore3270dsp).to_numpy()
x = np.arange(1,(len(dfcore190dsp) + 1),1)
l3 = ax1.plot(x, core390dsp, 'g-',  label = 'Core root 3 at 90 degrees')
l4 = ax1.plot(x, core3270dsp, 'k-',  label = 'Core root 3 at 270 degrees')
ax1.legend()
ax1.set_ylabel('D spacing')
ax1.set_xlabel('Frame number')
plt.show()


#core 1st order at different angular positions
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('D spacing per frame for skin 1st order sample 7 cooling')
skin10dsp = pd.DataFrame(dfskin10dsp).to_numpy()
skin1180dsp = pd.DataFrame(dfskin1180dsp).to_numpy()
y = np.arange(1,(len(dfskin10dsp) + 1),1)
l1 = ax1.plot(y, skin10dsp, 'b-', label = 'Skin 1st order at 0 degrees')
l2 = ax1.plot(y, skin1180dsp, 'r-', label = 'Skin 1st order at 180 degrees')
ax1.legend()
ax1.set_ylabel('D spacing')
ax1.set_xlabel('Frame number')
plt.show()


#core 1st order at different angular positions
fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize=(12,8))
fig1.suptitle('D spacing per frame for skin root 3 sample 7 cooling')

skin30dsp = pd.DataFrame(dfskin30dsp).to_numpy()
skin3180dsp = pd.DataFrame(dfskin3180dsp).to_numpy()
y = np.arange(1,(len(dfskin10dsp) + 1),1)
l3 = ax1.plot(y, skin30dsp, 'g-',  label = 'Skin root 3 at 0 degrees')
l4 = ax1.plot(y, skin3180dsp, 'k-',  label = 'Skin root 3 at 180 degrees')
ax1.legend()
ax1.set_ylabel('D spacing')
ax1.set_xlabel('Frame number')
plt.show()