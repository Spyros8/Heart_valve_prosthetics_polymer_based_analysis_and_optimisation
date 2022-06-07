# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:44:09 2021

@author: Arisa
"""
X=[]
Y=[]
lambd = 0.8856E-9
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
import math
from scipy import integrate
import statistics as stats
from openpyxl import Workbook
from matplotlib.pyplot import figure



def find_name(Name):
    X=[]
    Y=[]
    for subdir, dirs, files in os.walk(Name):
        for filename in files:
            filepath = subdir + os.sep + filename
            
           
        
            f=open(filepath,"r")
            
            fn={}
          
            x=[]
            y=[]
            
            
            for line in f:
                k,v = line.split()
                k=float(k)
                v=float(v)
                fn[k]=v
                x.append(k)
                y.append(v)
                
            x=list(x)
            y=list(y)
            Y.append(y)
            X.append(x)
    return(X,Y)




def peaks(array):
    Peaks=[]
    a=0
    for sample in array:
        fr=Int[a]
        Peaks.append(find_peaks(fr,threshold=0))
        a=a+1
    return(Peaks)

def order(Peaks,Angle,d,fr):
    
    N=[]
    
    
    a=0
    for item in Peaks:
        n=[]
        x=Peaks[a]
        x=list(x[0])
        
        i=0
        for index in x:
            ds=d[a][np.argmax(fr[a])]
            n.append([2*ds*10**(-9)*math.sin(math.radians(Angle[a][x[i]]/2))/lambd,index])
            i=i+1
            
        a=a+1
        n=list(n)
        
        N.append(n)
      
        
       
    return(N)

#Let's now try and keep the ones we want
def Main(factor,N):
    a=0
    order=[]
    
    first=[]
    second=[]
    root3=[]
    Int1=[]
    Introot3=[]
    Intsecond=[]
    first_1=[]
    for frame in N:
        first=[]
        second=[]
        root3=[]
        
        for pair in frame:
            
            peak=[]
            index=[]
            peak = pair[0]
            index = pair[1]
            
           
            if peak > 1*(1-factor) and peak< 1*(1+factor):
                first.append([peak,index])
                
            elif peak > math.sqrt(3)*(1-factor) and peak < math.sqrt(3)*(1+factor):
                root3.append([peak,index])
                
            elif   peak> 2*(1-factor) and peak<2*(1+factor):
                second.append([peak,index])
                
       
        order.append([first,root3,second]) 
        
        
            
            
        first = np.array(first)
        
        
        x = Angle[a][int(min(first[:,1])):int(max(first[:,1]))]
        y= Int[a][int(min(first[:,1])):int(max(first[:,1]))]
        Int1.append(integrate.trapz(y,x))
        
        #do the root3
        
        if not root3:
             Introot3.append(0)
        elif len(root3)==1:
            x=Angle[a][root3[0][1]-1],Angle[a][root3[0][1]],Angle[a][root3[0][1]+1]
            y=[1,2,3]
            Introot3.append(integrate.trapz(y,x))
            
        else:
             root3=np.array(root3)
             x = Angle[a][int(min(root3[:,1])):int(max(root3[:,1]))]
             y= Int[a][int(min(root3[:,1])):int(max(root3[:,1]))]
             Introot3.append(integrate.trapz(y,x))
        
        
        if not second:
            Intsecond.append(0)
        elif len(second)==1:
            
            Intsecond.append(Int[a][second[0][1]])
        else:
             second=np.array(second)
             x = Angle[a][int(min(second[:,1])):int(max(second[:,1]))]
             y= Int[a][int(min(second[:,1])):int(max(second[:,1]))]
             Intsecond.append(integrate.trapz(y,x))
       
        a=a+1
    return(Int1,Introot3,Intsecond,order)

    
    
def Fraction(Introot3):   
    centdiff=[]
    for value in Introot3:
        if Introot3.index(value)>1 and Introot3.index(value)< len(Introot3)-1:
            centdiff.append(abs((Introot3[Introot3.index(value)+1]-Introot3[Introot3.index(value)-1])/2))
    
    Sum=0
    it=0
    for value in Introot3:
        if Introot3.index(value)>1 and Introot3.index(value)< len(centdiff):
            if (Introot3.index(value)+1 -value)/value < 0.1 and value<max(Introot3)/2:
                Sum = Sum+ value
                it = it+1
                
    avg = Sum/it
    
    indeces = find_peaks(Introot3,height=avg*1.5)
    
    Fract = (max(indeces[0])-min(indeces[0]))/len(Introot3)
    return Fract,indeces


def dspace(d,indeces,fr):
    ds=[]
    for i in range(len(d)):
        if max(fr[i])==0:
            ds.append(0)
        else:
            
            ds.append(d[i][np.argmax(fr[i])])
    dskin=[]
    dcore=[]
    for i in range(len(ds)):
        if i < indeces[0] or i>indeces[1]:
            dskin.append(ds[i])
        else:
            dcore.append(ds[i])
    if indeces[0]==0 and indeces[1]==0:
        dcm=0
        dct=0
    else:
        
        dcm=stats.mean(dcore)
        dct=stats.stdev(dcore)
        
    dsm = stats.mean(dskin)
    dst=stats.stdev(dskin)
    return dcm,dct,dsm,dst,ds


def cut(entry,first,second):
    for i in range(first):
        entry.remove(entry[i])
    
    for i in range(second):
        entry.pop()
  
   
    return entry



    
first=17
second=14
file = "650"
name_angle= r"D:\Data\Processed_Data\Sample 5\Angle\i22-278"+file+"_ascii"
name_dspace= r"D:\Data\Processed_Data\Sample 5\dspace\i22-278"+file+"_ascii"


# Angle,Int = find_name(name_angle)
# Angle = cut(Angle,first,second)
# Int=cut(Int,first,second)
# d,fr = find_name(name_dspace)
# d=cut(d,first,second)
# fr=cut(fr,first,second)
# Peaks=peaks(Int)
# N=order(Peaks,Angle,d,fr)
# factor =0.1
# Int1,Introot3,Intsecond,order = Main(factor,N)
# #Fract,indeces=Fraction(Introot3)


# index=[20,25]
# dcm,dcdev,dsm,dsdev,ds = dspace(d,index)
# result =[dcm,dcdev,dsm,dsdev]
file1="517"
file2="525"
file3="534"
file4="546"

Sample="9"



d1,fr1=find_name(r"D:\Data\Processed_Data\Sample "+Sample+"\dspace\i22-278"+file1+"_ascii")
d2,fr2=find_name(r"D:\Data\Processed_Data\Sample "+Sample+"\dspace\i22-278"+file2+"_ascii")
d3,fr3=find_name(r"D:\Data\Processed_Data\Sample "+Sample+"\dspace\i22-278"+file3+"_ascii")
d4,fr4=find_name(r"D:\Data\Processed_Data\Sample "+Sample+"\dspace\i22-278"+file4+"_ascii")



first1=6
second1=2
first2=7
second2=8
first3=5
second3=11
first4=5
second4=0




indeces1=[14,55]
indeces2=[17,48]
indeces3=[23,40]
indeces4=[24,39]



d1=cut(d1,first1,second1)
d2=cut(d2,first2,second2)
d3=cut(d3,first3,second3)
d4=cut(d4,first4,second4)


a,b,c,d,ds1=dspace(d1,indeces1,fr1)
a,b,c,d,ds2=dspace(d2,indeces2,fr2)
a,b,c,d,ds3=dspace(d3,indeces3,fr3)
a,b,c,d,ds4=dspace(d4,indeces4,fr4)

fig=plt.figure()
plt.plot(range(len(ds1)),ds1,'tab:orange',label='12')
plt.plot(range(len(ds2)),ds2,'tab:blue',label='27')
plt.plot(range(len(ds3)),ds3,'tab:red',label='42')
plt.plot(range(len(ds4)),ds4,'tab:green',label='perpendicular')
plt.xlabel(xlabel='frame number')
plt.ylabel(ylabel= r'd($\AA$)')
fig.set_size_inches(15,8)
plt.legend()
plt.savefig("All_Sample"+Sample+".png")
plt.show()

fig2=plt.figure()
plt.plot(range(len(ds1)),ds1,'tab:blue')
fig2.set_size_inches(15,8)
plt.title("Most frequent d-spacing, 12 mm down sample " + Sample)
plt.savefig(file1+".png")
plt.xlabel(xlabel='frame number')
plt.ylabel(ylabel= r'd($\AA$)')
plt.show()

fig3=plt.figure()
plt.plot(range(len(ds2)),ds2,'tab:blue')
fig3.set_size_inches(15,8)
plt.title("Most frequent d-spacing, 27 mm down sample " + Sample)
plt.xlabel(xlabel='frame number')
plt.ylabel(ylabel= r'd($\AA$)')
plt.savefig(file2+".png")
plt.show()

fig4=plt.figure()
plt.plot(range(len(ds3)),ds3,'tab:blue')
fig4.set_size_inches(15,8)
plt.title("Most frequent d-spacing, 42 mm down sample " + Sample)
plt.xlabel(xlabel='frame number')
plt.ylabel(ylabel= r'd($\AA$)')
plt.savefig(file3+".png")
plt.show()

fig5=plt.figure()
plt.plot(range(len(ds4)),ds4,'tab:blue')
fig5.set_size_inches(15,8)
plt.xlabel(xlabel='frame number')
plt.ylabel(ylabel= r'd($\AA$)')
plt.title("Most frequent d-spacing, 42 mm down sample " + Sample+"(perpendicular)")
plt.savefig(file4+".png")
plt.show()













def fuck():

    first=[]
    root3=[]
    second=[]
    Integral_first=[]
    Integral_Second=[]
    Integral_Root3=[]
    
    
    for i in range(len(Angle)):
        Frame=Angle[i]
        n=[]
        first=[]
        root3=[]
        second=[]
        firstindex=[]
        root3index=[]
        secondindex=[]
        Inten1=0
        Intenroot3=0
        Inten2=0
        for a in range(len(Frame)):
            n.append([2*ds[i]*10**(-9)*math.sin(math.radians(Angle[i][a]/2))/lambd])
            z=0
        for peak in n:
            
            peak=float(peak[0])
            if peak > 1*(1-factor) and peak< 1*(1+factor):
                first.append(peak)
                firstindex.append(z)
                
                
            elif peak > math.sqrt(3)*(1-factor) and peak < math.sqrt(3)*(1+factor):
                root3.append(peak)
                root3index.append(z)
                
            elif   peak> 2*(1-factor) and peak<2*(1+factor):
                second.append(peak)
                secondindex.append(z)
            z=z+1
        Inten1=Int[i][min(firstindex):max(firstindex)]
        Ang=Angle[i][min(firstindex):max(firstindex)]
        Integral_first.append(integrate.trapz(Inten1,Ang))
        
        Intenroot3=Int[i][min(root3index):max(root3index)]
        Ang3=Angle[i][min(root3index):max(root3index)]
        Integral_Root3.append(integrate.trapz(Intenroot3,Ang3))
        
        Inten2=Int[i][min(secondindex):max(secondindex)]
        Ang2=Angle[i][min(secondindex):max(secondindex)]
        Integral_Second.append(integrate.trapz(Inten2,Ang2))
    count=[]
    Sum=0   
    for i in range(len(Integral_Root3)):
        if Integral_Root3[i]<0.65*max(Integral_Root3):
            count.append(0)
        else:
            count.append(1)
            Sum=Sum+1
                    
    print(Sum/len(count))





