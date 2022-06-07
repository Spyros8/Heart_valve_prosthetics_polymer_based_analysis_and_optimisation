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
from scipy.signal import peak_widths
from scipy.stats import rankdata
from scipy.signal import find_peaks_cwt

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
# This function subtracts the background from each value in the array. The input array should be a list of each data frame in dawn, containing all the values for each frame ie
# ie array[1][1] is the first element of the first data frame of the sample. The returned array has the same data structure
def background(Array,background):
    b=[]
    for i in range(len(Array)):
        a=[]
        for j in range(len(Array[i])):
            if Array[i][j]-background <0:
                a.append(0)
            else:
                a.append(Array[i][j]-background)
            
        b.append(a)
    return(b)


def centdiff(Array):
    b=[]
    for i in range(len(Array)):
        a=[]
        for j in range(len(Array[i])):
            if j>0 and j< len(Array[i])-1:
                
                a.append((Array[i][j+1]-Array[i][j-1])/2)
            elif j==0:
                a.append(Array[i][j+1]-Array[i][j])
            elif j== len(Array[i])-1:
                a.append(Array[i][j]-Array[i][j-1])
        b.append(a)
    return(b)

def cleanup(Array,threshold):
    b=[]
    for i in range(len(Array)):
        a=[]
        for j in range(len(Array[i])):
            if abs(Array[i][j])<threshold:
                a.append(0)
            else:
                a.append(Array[i][j])
        b.append(a)
    return(b)

def FMHW(Array,rel_height):
    left=[]
    right=[]
    Peaks=[]
    for i in range(len(Array)):
        
        peaks=0
        Widths=0
        peaks = find_peaks(Array[i])
        Widths = peak_widths(Array[i],peaks[0],rel_height=rel_height)
        left.append(Widths[2])
        right.append(Widths[3])
        Peaks.append(peaks[0])
    d=[]
    for i in range(len(left)):
        diff=[]
        if not any(left[i]):
            pass
        else:
            for j in range(len(left[i])):
                if right[i][j]-left[i][j]>5:
                    diff.append(right[i][j]-left[i][j])
                else:
                    diff.append(0)
        d.append(diff)
    return(left,right,d,Peaks)

def FMHW_noise(Array,rel_height,height):
    left=[]
    right=[]
    Peaks=[]
    for i in range(len(Array)):
        
        peaks=0
        Widths=0
        peaks = find_peaks(Array[i],height=height)
        Widths = peak_widths(Array[i],peaks[0],rel_height=rel_height)
        left.append(Widths[2])
        right.append(Widths[3])
        Peaks.append(peaks[0])
    d=[]
    for i in range(len(left)):
        diff=[]
        if not any(left[i]):
            pass
        else:
            for j in range(len(left[i])):
                if right[i][j]-left[i][j]>5:
                    diff.append(right[i][j]-left[i][j])
                else:
                    diff.append(0)
        d.append(diff)
    return(left,right,d,Peaks)


def FMHW_sample(Array,rel_height):
    left=[]
    right=[]
    Peaks=[]
    for i in range(len(Array)):
        
        peaks=0
        Widths=0
        peaks = find_peaks(Array[i],distance=20)
        it=0
        distance = 20
        
        while len(peaks[0])>5 and it<500:
    
            it=it+1
            peaks = find_peaks(Array[i], distance = distance)
            distance = distance+5
        
    
        
        Widths = peak_widths(Array[i],peaks[0],rel_height=rel_height)
        left.append(Widths[2])
        right.append(Widths[3])
        Peaks.append(peaks[0])
    d=[]
    for i in range(len(left)):
        diff=[]
        if not any(left[i]):
            pass
        else:
            for j in range(len(left[i])):
                if right[i][j]-left[i][j]>5:
                    diff.append(right[i][j]-left[i][j])
                else:
                    diff.append(0)
        d.append(diff)
    return(left,right,d,Peaks)
        
def FMHW2(Array, left, right, position,peaks):
    a=[]
    posi=[]
    Left=[]
    Right=[]
    P=[]
    for i in range(len(Array)):
        b=[]
        L=[]
        R=[]
        pos=[]
        p=[]
        for j in range(len(Array[i])):
            if Array[i][j] ==0:
                
                pass
            else:
                r=round(right[i][j])
                l=round(left[i][j])
                pos.append([position[i][l],position[i][r]])
                b.append(position[i][r]-position[i][l])
                L.append(l)
                R.append(r)
                p.append(peaks[i][j])
                
                
        a.append(b)
        posi.append(pos)
        Left.append(L)
        Right.append(R)
        P.append(p)
    return(a,posi,Left,Right,P)
            
                
def Noise(Array_position,Noise_position,Intensity,Noise,error):
   
    Peaks=[]
    for i in range(len(Array_position)):
        peak=[]
        for k in range(len(Array_position[i])):
            
            cont=0
            for j in range(len(Noise_position[i])):
                
                l_n = round(Noise_position[i][j][0])
                r_n=round(Noise_position[i][j][1])
                
                if (Array_position[i][k][0] in range(l_n,r_n) or Array_position[i][k][1] in range(l_n,r_n)) and Noise[i][round((l_n+r_n)/2)]>error*Intensity[i][round((Array_position[i][k][0]+Array_position[i][k][1])/2)]:
                    cont=cont+1
                    
            if cont>1:
                pass
            else:
                peak.append(Array_position[i][k])
        Peaks.append(peak)                
                
    return(Peaks)

def rank(Array,lefty, Angle):
    Sample=[]
    for i in range(len(Array)):
        frame=[]
        print(i)
        for j in range(len(Array[i])):
            
                
            Intensity = []
            Peak=[]
            if len(lefty[i])==1:
                    left = round(lefty[i][0])
                    Peak.append(left)
            elif any(lefty[i]) == False:
                Peak.append([])
            else:
                for k in range(len(lefty[i])):
                
                
                     left = round(lefty[i][k])
                
                
                
                
                     Intensity.append(Array[i][left])
                ranked_intensity = Intensity.sort() 
                
                for a in range(len(ranked_intensity)):
                    index1 = Intensity.index(ranked_intensity[a])
                    Peak.append([left[i][index1][0],Intensity])
            frame.append(Peak)
        Sample.append(frame)
    return(Sample)

def Background_Noise(Noise, left, right, Angle):
    Avg=[]
    for i in range(len(Noise)):
        Average=0
        
        Sum=0
        Peak_frames=0
        for k in range(len(left[i])):
            l = round(left[i][k])
            r = round(right[i][k])
            x = Angle[i][l:r+1]
            y =Noise[i][l:r+1]
            Sum = Sum+integrate.trapz(y,x)
            Peak_frames=Peak_frames+(Angle[i][r]-Angle[i][l])
        Total = integrate.trapz(Noise[i],Angle[i])
        Residual = Total-Sum
        Average = Residual/(Angle[i][-1]-Peak_frames)
        Avg.append(Average)
        
    return(Avg)

            
def background_frame(Array,background):
    b=[]
    for i in range(len(Array)):
        a=[]
        for j in range(len(Array[i])):
            if Array[i][j]-background[i] <0:
                a.append(0)
            else:
                a.append(Array[i][j]-background[i])
            
        b.append(a)
    return(b)
            

                                
             
            
    
                
            
            
    
        
        
        
                
Angle,Int = find_name(r"E:\data\Raw Data\Processes stuff\First Order\First Order\All\i22-272743-Pilatus2M_SAXS-002_ascii")
Angle_n, Int_n = find_name(r"E:\data\Raw Data\Processes stuff\First Order\Noise\All\i22-272743-Pilatus2M_SAXS-002_ascii")
l,r,diffe,peaks=FMHW_sample(Int,rel_height=0.5)
l_n,r_n,diffe_n,peaks_n = FMHW(Int_n,rel_height=1)
back=Background_Noise(Int_n,l_n,r_n,Angle_n)
a= background_frame(Int,back)
a_n = background_frame(Int_n,back)






l,r,diffe,peaks=FMHW_sample(a,rel_height=0.5)
l_n,r_n,diffe_n,peaks_n = FMHW_noise(a_n,rel_height=1,height=0)
p,posi,left,right,peaks_f=FMHW2(diffe,l,r,Angle,peaks)
p_n,posi_n,left_n,right_n,peaks_f_n=FMHW2(diffe_n,l_n,r_n,Angle_n,peaks_n)


                                 

pls=Noise(posi,posi_n,Intensity=a, Noise=a_n,error =0.25)



    







def Pick_Peak(Peaks,Angle,Intensity,left,right):
    Right_peak_left=[]
    Right_peak_right=[]
    Right_peak_index=[]
    
    Right_peak_index=[]
    New_array=[]
    New_angle=[]
    Peak_index=[]
    Skin_Integral=[]
    
    for i in range(len(Peaks)):
        Right_peak=[]
       
        if any(Peaks[i]):
            peak_index=[]
            for j in range(len(Peaks[i])):
                
                Angular_pos = round(Angle[i][Peaks[i][j]])
               
                            
                if Angular_pos in range(45,135) or Angular_pos in range(225,315):
                    peak_index.append([Peaks[i][j], "Core"])
                elif Angular_pos in range(315,360) or Angular_pos in range(0,45):
                    Right_peak.append(Peaks[i][j])
                elif Angular_pos in range(135,225):
                    peak_index.append([Peaks[i][j], "Skin"])
                    
            if len(Right_peak)>1:
                left_index = left[i][-1] # This relies on the peaks being ordered from smallest to highest value
                right_index = right[i][0]
                Right_peak_index.append([left_index,right_index])
                New_new_array=[]
                New_new_angle=[]
                for k in range (left_index-100,len(Intensity[i])):
                    New_new_array.append(Intensity[i][k])
                    New_new_angle.append(Angle[i][k])
                Int1= integrate.trapz(New_new_array, New_new_angle)    
                for m in range(2,right_index+100):
                    New_new_array.append(Intensity[i][m])
                    New_new_angle.append(Angle[i][m]+360)
                Int2= integrate.trapz(New_new_array[k-left_index+100+1:-1], New_new_angle[k-left_index+100+1:-1])
                Int = Int1+Int2
               
            
                Skin_Integral.append(Int)
                New_array.append(New_new_array)
                New_angle.append(New_new_angle)
                peaks_new=0
                Widths=0
                peaks = find_peaks(New_array[i],distance=20)
                it=0
                distance = 20
                
                while len(peaks[0])>1 and it<500:
            
                    it=it+1
                    peaks_new = find_peaks(New_array[i], distance = distance)
                    distance = distance+5
                
            
                
                Widths = peak_widths(New_array[i],peaks_new[0],rel_height=0.5)
                
                
                Right_peak_left.append(np.round(Widths[2][0]))
                Right_peak_right.append(np.round(Widths[3][0]))
                
                if New_angle[i][round(peaks_new[0][0])]<180:
                    peak_index.append([round(peaks_new[0][0])-(len(Intensity[i])-(left_index-100)),"Skin"])
                else:
                    peak_index.append([round(peaks_new[0][0])-left_index-100,len(Intensity[i]), "Skin"])
                    
                    
                
            elif len(Right_peak)==1:
                peak_index.append([Peaks[i][j], "Skin"])
                New_array.append([])
                New_angle.append([])
                Right_peak_left.append([])
                Right_peak_right.append([])
                Right_peak_index.append([])
                Skin_Integral.append([])
            else:
                New_array.append([])
                New_angle.append([])
                Right_peak_left.append([])
                Right_peak_right.append([])
                Right_peak_index.append([])
                Skin_Integral.append([])
            Peak_index.append(peak_index)   
        else:
            Peak_index.append([])
            New_array.append([])
            New_angle.append([])
            Right_peak_left.append([])
            Right_peak_right.append([])
            Right_peak_index.append([])
            Skin_Integral.append([])
            
            
    return(Peak_index,New_array,New_angle,Right_peak_left,Right_peak_right,Right_peak_index,Skin_Integral)
        
                
peak_index,New_array,New_angle,Right_peak_left,Right_peak_right,Right_peak_index,Skin_Integral=Pick_Peak(peaks_f,Angle,a,left,right)

def Integrals(Peaks, Angle, Intensity,Skin_Integral,Right_peak_left,Right_peak_right,New_angle):
    Skin_Intensity=[]
    Core_Intensity=[]
    FHW_Sample=[]
    for i in range(len(Peaks)):
        Core_Int=0
        Skin_Int=0
        FHW_angle_frame=[]
        for j in range(len(Peaks[i])):
            
            
            Peak =[Peaks[i][j][0]]
            
            if Peaks[i][j][1] == "Core":
                width = peak_widths(Intensity[i],Peak,rel_height = 0.95)
                FHW = peak_widths(Intensity[i],Peak,rel_height = 0.5)
                FHW_angle = Angle[i][int(np.round(FHW[3][0]))] - Angle[i][int(np.round(FHW[2][0]))]
                x = Angle[i][int(round(width[2][0])):int(round(width[3][0]))] # This only works if Peaks[i][j][0] has only one peak
                y = Intensity[i][int(round(width[2][0])):int(round(width[3][0]))]
                Core_Int = Core_Int + integrate.trapz(y,x)
                FHW_angle_frame.append([Peaks[i][j][0],FHW_angle,"Core"])
            elif Peaks[i][j][1] == "Skin":
              
                if Angle[i][Peaks[i][j][0]]>270 or  Angle[i][Peaks[i][j][0]]< 45:
                    
                    if type(Skin_Integral)==list:
                        width = peak_widths(Intensity[i],Peak,rel_height = 0.95)
                        x = Angle[i][int(round(width[2][0])):int(round(width[3][0]))] # This only works if Peaks[i][j][0] has only one peak
                        y = Intensity[i][int(round(width[2][0])):int(round(width[3][0]))]
                        Skin_Int = Skin_Int + integrate.trapz(y,x)
                        FHW = peak_widths(Intensity[i],Peak,rel_height = 0.5)
                        FHW_angle = Angle[i][int(np.round(FHW[3][0]))] - Angle[i][int(np.round(FHW[2][0]))]
                        FHW_angle_frame.append([Peaks[i][j][0],FHW_angle,"Skin"])
                      
                      
                    else:
                        Skin_Int = Skin_Int + Skin_Integral[i]
                        for h in range(len(New_angle[i])):
                           
                            if New_angle[i][j]<360:
                                New_angle[i][j]=New_angle[i][j]+360
                                FHW_angle = New_angle[i][Right_peak_right[i]]-New_angle[i][Right_peak_left[i]]
                                FHW_angle_frame.append([Peaks[i][j][0],FHW_angle,"Skin"])
                            
                            
                        
                        
                else:
                    width = peak_widths(Intensity[i],Peak,rel_height = 0.95)
                    x = Angle[i][int(round(width[2][0])):int(round(width[3][0]))] # This only works if Peaks[i][j][0] has only one peak
                    y = Intensity[i][int(round(width[2][0])):int(round(width[3][0]))]
                    Skin_Int = Skin_Int + integrate.trapz(y,x)
                    FHW = peak_widths(Intensity[i],Peak,rel_height = 0.5)
                    FHW_angle = Angle[i][int(np.round(FHW[3][0]))] - Angle[i][int(np.round(FHW[2][0]))]
                    FHW_angle_frame.append([Peaks[i][j][0],FHW_angle,"Skin"])
        Skin_Intensity.append(Skin_Int)
        Core_Intensity.append(Core_Int)
        FHW_Sample.append(FHW_angle_frame)
        
    return(Core_Intensity,Skin_Intensity,FHW_Sample)
                    
Core_Intensity,Skin_Intensity, FHW_sample = Integrals(peak_index, Angle, a,Skin_Integral,Right_peak_left,Right_peak_right,New_angle)                    




# D:\data\Raw Data\Processes stuff\First Order\First Order\All
# D:\data\Raw Data\Processes stuff\First Order\Noise\All



Entries = ["\i22-272742-Pilatus2M_SAXS-002_ascii","\i22-272743-Pilatus2M_SAXS-001_ascii"]
Entries.append("\i22-272750-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272751-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272752-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272753-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272760-Pilatus2M_SAXS_ascii")
#Entries.append("\i22-272762-Pilatus2M_SAXS_ascii")
#Entries.append("\i22-272763-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272764-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272765-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272766-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272767-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272768-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272769-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272770-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272771-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272772-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272773-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272774-Pilatus2M_SAXS_ascii")
#Entries.append("\i22-272775-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272779-002-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272780-001-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272786-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272783-Pilatus2M_SAXS-002_ascii")
Entries.append("\i22-272784-Pilatus2M_SAXS-001_ascii")
#Entries.append("\i22-272787-Pilatus2M_SAXS_ascii")
#Entries.append("\i22-272788-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272789-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272790-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272792-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272793-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272794-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272795-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272820-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272821-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272822-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272823-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272824-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272825-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272826-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272827-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272828-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272829-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272830-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272831-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272832-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272833-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272834-Pilatus2M_SAXS_ascii")
Entries.append("\i22-272835-Pilatus2M_SAXS_ascii")

All_Data=[["Core Intensity", "Skin Intensity",  "Core FHW", "Skin FHW"]]

for name in Entries:
    print(name)
    
    Angle,Int = find_name(r"E:\data\Raw Data\Processes stuff\First Order\First Order\All" + name)
    Angle_n, Int_n = find_name(r"E:\data\Raw Data\Processes stuff\First Order\Noise\All" + name)
    
    l,r,diffe,peaks=FMHW_sample(Int,rel_height=0.5)
    l_n,r_n,diffe_n,peaks_n = FMHW(Int_n,rel_height=1)
    back=Background_Noise(Int_n,l_n,r_n,Angle_n)
    a= background_frame(Int,back)
    a_n = background_frame(Int_n,back)
    
    
    
    
    
    
    l,r,diffe,peaks=FMHW_sample(a,rel_height=0.5)
    l_n,r_n,diffe_n,peaks_n = FMHW_noise(a_n,rel_height=1,height=0)
    p,posi,left,right,peaks_f=FMHW2(diffe,l,r,Angle,peaks)
    p_n,posi_n,left_n,right_n,peaks_f_n=FMHW2(diffe_n,l_n,r_n,Angle_n,peaks_n)
    
    
                                     
    
    pls=Noise(posi,posi_n,Intensity=a, Noise=a_n,error =0.25)
    peak_index,New_array,New_angle,Right_peak_left,Right_peak_right,Right_peak_index,Skin_Integral=Pick_Peak(peaks_f,Angle,a,left,right)
    Core_Intensity,Skin_Intensity, FHW_sample = Integrals(peak_index, Angle, a,Skin_Integral,Right_peak_left,Right_peak_right,New_angle)
    
    Final_Core_list=[]
    Final_Skin_list=[]
    for i in range(len(FHW_sample)):
        Core_list=0
        Skin_list=0
        count=0
        counts=0
        for j in range(len(FHW_sample[i])):
            
            if FHW_sample[i][j][2] == "Core":
                Core_list=Core_list + FHW_sample[i][j][1]
                count=count+1
            elif FHW_sample[i][j][2 == "Skin"]:
                Skin_list=Skin_list + FHW_sample[i][j][1]
                counts=counts+1
        if count == 0:
            Final_Core_list.append(0)
            
        else:
            
            Final_Core_list.append(Core_list/count)
            
        if counts==0:
            Final_Skin_list.append(0)
        else:
            Final_Skin_list.append(Skin_list/counts)
    All_Data.append([Core_Intensity,Skin_Intensity, Final_Core_list,Final_Skin_list])
    
    fig=plt.figure()
    plt.plot(range(len(Core_Intensity)),Core_Intensity, 'r.', label='Core')
    plt.plot(range(len(Skin_Intensity)),Skin_Intensity, 'b.', label='Skin')
    plt.title('Skin and Core Intensities for sample'+name[8:11])
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('Intensity')
    fig.set_size_inches(15,8)
    plt.savefig("Int_"+name[8:11]+".png")
    plt.close()
    
    
    fig2=plt.figure()
    plt.plot(range(len(Final_Core_list)),Final_Core_list, 'r.', label='Core')
    plt.plot(range(len(Final_Skin_list)),Final_Skin_list, 'b.', label='Skin')
    plt.title('Skin and Core Full Width Half Maxima for sample' +name[8:11])
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('Intensity')
    fig2.set_size_inches(15,8)
    plt.savefig("FHA_"+name[8:11]+".png")
    plt.close()
    
    
    
    
    
            
    

                    
                                


# fig=plt.figure()
# plt.plot(range(len(ds1)),ds1,'tab:orange',label='12')
# plt.plot(range(len(ds2)),ds2,'tab:blue',label='27')
# plt.plot(range(len(ds3)),ds3,'tab:red',label='42')
# plt.plot(range(len(ds4)),ds4,'tab:green',label='perpendicular')
# plt.savefig("All_+".png")
# fig.set_size_inches(15,8)

# plt.legend()
# plt.show()

# fig2=plt.figure()
# plt.plot(range(len(ds1)),ds1,'tab:blue')
# fig2.set_size_inches(15,8)
# plt.title("Most frequent d-spacing, 12 mm down sample " + Sample)
# plt.legend()
# plt.savefig(file1+".png")
# plt.show()

# fig2=plt.figure()
# plt.plot(range(len(ds2)),ds2,'tab:blue')
# fig2.set_size_inches(15,8)
# plt.legend()
# plt.title("Most frequent d-spacing, 27 mm down sample " + Sample)
# plt.savefig(file2+".png")
# plt.show()

# fig2=plt.figure()
# plt.plot(range(len(ds3)),ds3,'tab:blue')
# fig2.set_size_inches(15,8)
# plt.legend()
# plt.title("Most frequent d-spacing, 42 mm down sample " + Sample)
# plt.savefig(file3+".png")
# plt.show()

# fig2=plt.figure()
# plt.plot(range(len(ds4)),ds4,'tab:blue')
# fig2.set_size_inches(15,8)
# plt.legend()
# plt.title("Most frequent d-spacing, 42 mm down sample " + Sample+"(perpendicular)")
# plt.savefig(file4+".png")
# plt.show()
                    
                    
                
                
                
       
            
                
                      
                 


# # def Pick_Peak(Array,left,right,Angle,error):
# Array = a


# Angle=Angle
# error =10

# indexi=[]
# for i in range(len(Array)):
#     # indexj=[]
#     print(i)
    
#     # for j in range(len(Array[i])):
#     #     index=[]
#     index=[]    
#     for k in range(len(left[i])):
        
#         left_test = round(left[i][k])
#         right_test = round(right[i][k])
#         for m in range(len(left[i])):        
                
#             if m==k:
                
#                 pass
#             else:
                
            
#                 left_interval = Angle[i][round(left[i][m])]
#                 right_interval=Angle[i][round(right[i][m])]
                
#                 if left_interval +180 > 360:
#                     left_interval = left_interval-180
#                 else:
#                     right_interval = right_interval +180
#                 if right_interval + 180 > 360:
#                     right_interval = right_interval-180
#                 else:
#                     right_interval = right_interval +180
                    
#                 if any(left[i]) == False:
#                     pass
#                 elif len(left[i])<4:
#                     if Angle[i][left_test] in range(310,360) or Angle[i][left_test] in range(0,40) or Angle[i][left_test] in range(130,220):
#                         index.append([[left_test,right_test], [], "Skin"])
#                     else:
#                         index.append([[left_test,right_test], [], "Core"])
#                 else:
                    
#                     if Angle[i][left_test] in range(round(left_interval-error),round(right_interval+error)) or Angle[i][right_test] in range(round(left_interval-error),round(right_interval+error)):
#                         if Angle[i][left_test] in range(310,360) or Angle[i][left_test] in range(0,40) or Angle[i][left_test] in range(130,220):
#                             index.append([[left_test,right_test], [left_interval,right_interval], "Skin"])
#                         else:
#                             index.append([[left_test,right_test], [left_interval,right_interval], "Core"])
                       
#         # indexj.append(index)
#     indexi.append(index)
#     # return(indexi)








#rank1=rank(Int,l,Angle)
# Background_n=[]
# for i in range(len(Int_n)):
#     it=0
#     backg=100
#     back_0=100
#     back_1=0
#     while -back_1+back_0>10 and it<500:
#         it=it+1
#         a_n= background(Int_n,back_1)
#         back_0 = back_1
#         backg = Background_Noise(Int_n,l_n,r_n,Angle_n)
#         back_1 = backg [7]
#         print(it)
#     Background_n.append(back_1)









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





