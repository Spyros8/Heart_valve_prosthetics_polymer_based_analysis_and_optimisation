import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from scipy.integrate import simps; from numpy import trapz

from mpl_toolkits import mplot3d
import seaborn as sns

# Safe to ignore warnings
SEPSFILES= 'C:/Users/Spyros/Downloads/skin and core fraction estimates new - CORRECTED.xlsx'
xl = pd.ExcelFile(SEPSFILES)
print(xl.sheet_names)

dfSEPS1 = xl.parse('Sheet1')
dfSEPS2 = xl.parse('Sheet2')
print(dfSEPS1.head())

sns.distplot(dfSEPS1['core fraction'])

sns.distplot(dfSEPS1['skin fraction'])

sns.pairplot(dfSEPS1,hue='thickness',palette='coolwarm')#hue for categorical
sns.pairplot(dfSEPS1) #joint plot for every combination of numerical plots in the dataframe
# Safe to ignore warnings

sns.pairplot(dfSEPS2, hue = 'thickness', palette = 'coolwarm')
sns.pairplot(dfSEPS2)



#trends with thickness
sns.jointplot(x='thickness',y='core fraction', data=dfSEPS1,kind='scatter', hue = 'Distance from injection point')

n=sns.jointplot(x='thickness',y='skin fraction', data=dfSEPS1,kind='scatter', hue = 'Distance from injection point')

n=sns.jointplot(x='cooling',y='skin fraction', data=dfSEPS1,kind='scatter', hue = 'thickness')
n.fig.suptitle('Parallel cuts')
sns.factorplot(x='thickness',y='skin fraction',data=dfSEPS1,kind='bar')

#sns.jointplot(x='thickness',y='core fraction total', data=dfSEPS2,kind='scatter', hue = 'annealing')

f=sns.jointplot(x='thickness',y='skin fraction total', data=dfSEPS2,kind='scatter', hue = 'annealing')
f.fig.suptitle('Perpendicular cuts')
p=sns.jointplot(x='annealing',y='skin fraction total', data=dfSEPS2,kind='scatter', hue = 'thickness')
p.fig.suptitle('Perpendicular cuts')

#pvdfSEPS2 = dfSEPS2.iloc[:, 1:5].pivot_table(values = dfSEPS1.iloc[:, 2:4],index='thickness',columns='annealing')
#sns.heatmap(pvdfSEPS2)


f=sns.jointplot(x='cooling',y='skin fraction total', data=dfSEPS2,kind='scatter', hue = 'thickness')
f.fig.suptitle('Perpendicular cuts')