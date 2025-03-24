import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
    
#%%
#Temperature Coeff Index:
    
#PA_78
#m_T = -4.74E-3

#PA_77
#m_T = -7.246355e-05

#combine 024 and 025
file = '250108/024_M81_PA.csv'
data1 = pd.read_csv(file)

file = '250108/025_M81_PA.csv'
data2 = pd.read_csv(file)

data = pd.concat([data1[:3680], data2[:3640]])
#%%
# file  = '230510_GaAs_D181211Ai_1V_1Mohm_LIA8_12_Rxx_Ryy_4K_p1T_m1T_Bsweep.csv'
#file = '241119/002_M81_B_ETH_F240417B_L5_n_AXT_Cr_QHE_2K.csv'
file = '250108/017_M81_PA.csv'

data = pd.read_csv(file)

# print(data)
I = 1000e-6
v = True  #True if data file recorded Rxx/Rxy/Ryx as the voltage signal

Rxx_X = np.array(data["Rxx_X[ohms]"], dtype=float)
Rxx_Y = np.array(data["Rxx_Y[ohms]"], dtype=float)

Rxy_X = np.array(data["Rxy_X[ohms]"], dtype=float)
Rxy_Y = np.array(data["Rxy_Y[ohms]"], dtype=float)
Ryx_X = np.array(data["Ryx_X[ohms]"], dtype=float)
Ryx_Y = np.array(data["Ryx_Y[ohms]"], dtype=float)

Rho_xy = (Rxy_X - Ryx_X) / (2)
Rho_xy_diff = (Rxy_X + Ryx_X) / (2)
T_Sample = np.array(data["T_Sample[K]"], dtype=float)
T_VTI = np.array(data["T_VTI[K]"], dtype=float)
B_Analog = np.array(data["Field_Analog[T]"], dtype=float)
Time = np.array(data["Elapsed Time[s]"])

#Freq = np.array(data["Freq_1[Hz]"], dtype=float)


if v == True:
    Vxy_X = Rxy_X
    Vxy_Y = Rxy_Y

    Vyx_X = Ryx_X
    Vyx_Y = Ryx_Y

    Vxx_X = Rxx_X
    Vxx_Y = Rxx_Y
    
    Rxy_X = Rxy_X/I
    Rxy_Y = Rxy_Y/I

    Ryx_X = Ryx_X/I
    Ryx_Y = Ryx_Y/I

    Rxx_X = Rxx_X/I
    Rxx_Y = Rxx_Y/I

    Rho_xy = (Rxy_X + Ryx_X) / (2)
    Rho_xy_diff = (Rxy_X - Ryx_X) / (2)


    

'''

#Concat datafiles

file = '241119/003_M81_B_ETH_F240417B_L5_n_AXT_Cr_QHE_2K.csv'
data = pd.read_csv(file)

Rxx_X_2 = np.array(data["Rxx_X[ohms]"], dtype=float)
Rxy_X_2 = np.array(data["Rxy_X[ohms]"], dtype=float)
Ryx_X_2 = np.array(data["Ryx_X[ohms]"], dtype=float)
Rho_xy_2 = (Rxy_X_2 - Ryx_X_2) / (2)
T_Sample_2 = np.array(data["T_Sample[K]"], dtype=float)
B_Analog_2 = np.array(data["Field_Analog[T]"], dtype=float)


dat_1 = [Rxx_X, Rxy_X, Ryx_X, Rho_xy, T_Sample, B_Analog]
dat_2 = [Rxx_X_2, Rxy_X_2, Ryx_X_2, Rho_xy_2, T_Sample_2, B_Analog_2]

for n in range(len(dat_1)):
    dat_1[n] = np.concatenate((dat_1[n], dat_2[n]), axis = 0)


Rxx_X = dat_1[0]
Rxy_X = dat_1[1]
Ryx_X = dat_1[2]
Rho_xy = dat_1[3]
T_Sample = dat_1[4]
B_Analog = dat_1[5]


'''

# plt.scatter(data.An_Field, data.Vxx_x, color = 'g')
# plt.scatter(data.An_Field, data.Vyy_x, color = 'b')

plt.scatter(B_Analog, Rxy_X, color = 'b', label = "Rxy", s = 0.1)
plt.scatter(B_Analog, Ryx_X, color = 'r', label = "Ryx", s = 0.1)
#plt.scatter(B_Analog, Ryx_Y, color = 'b', label = "Ryx_Y", s = 0.1)

#plt.scatter(B_Analog, Rho_xy, color = 'Orange', label = "Rhoxy", s= 0.1)
#plt.scatter(B_Analog, Rxx_X, color = 'g', label = "Rxx", s = 0.1)

#plt.plot(B_Analog, T_Sample, color = 'r')
#plt.plot(B_Analog, data.Vyx_x/I, color = 'b')

#plt.plot(data.An_Field, data.Vxy_x, color = 'g')
#plt.plot(data.An_Field, data.Vyx_x, color = 'b')




#%%
#Linear fit

#001: [:6941]
#002: [:7000]
#003: [:6930]
#004: None
#005: [:2480]
#006: [:5850]
#007: [:2480]
#008: [:2510]
#012: [:790]
#017: [:10830]
#018: [:12450], [10680:12226]
#019: [:5735], [:2820], [2820:5737]
#022: [:10820], below 6.5T = [:4850]
#024: [:3680]
#025 + 024: [:3680], [4880:8490]
#026: [:7250]
#028: [:1560]

#030: 2T [:1538] Blowup [:5150]



#Si doped with a spacer thickness of...
#SEND 10mA, 
plot_V = False  #True, plot Vxy, Vyx assuming dataset input is V. False plot Rxy

plot_Freq = False 
plot_B = 1 #x_axis t if True, B if false

plot_fit = 1
fit_range = 10830

#[:3020]
#[5800:]
#[160:12360]
#[10680:12226]
#

timeline = False
#009
#Noise beat of ~28s, 0.0356Hz

#010
slices = [0, 220, 470, 870, 925, 1410, 1790, 2400, 3180, 3900, 4660, 5260]
#011
slices = [0, 1340, 2270, 9000]
#Noise beat of ~211s, .00474HZ

#014
slices = [0, 3270, 5615]

#016
slices = [0, 1730, 3930]


timeline_ylim = [-3, 3]

if plot_fit == True:
    
    #m, b = np.polyfit(B_Analog[:fit_range], Rxy_X[:fit_range], 1)
    #m, b = np.polyfit(B_Analog[fit_range:], Rxy_X[fit_range:], 1)
    
    
    #m1, m2, c = np.polyfit(B_Analog[:fit_range], Rxy_X[:fit_range], 2)
    #m1, m2, c = np.polyfit(B_Analog[:fit_range], Ryx_X[:fit_range], 2)
    #m1, m2, c = np.polyfit(B_Analog[:fit_range], Rho_xy[:fit_range], 2)
    m1, m2, c = np.polyfit(B_Analog[:fit_range], Rho_xy_diff[:fit_range], 2)

    #m, b = np.polyfit(B_Analog[10680:12226], Rxy_X[10680:12226], 1)


plt.figure()
if plot_B == True:
    if plot_V == True:
        plt.scatter(B_Analog, Vxy_X, color = 'b', label = "Vxy", s = 0.1)
        plt.scatter(B_Analog, Vyx_X, color = 'r', label = "Vyx", s = 0.1)
    
    else:
        
        #plt.scatter(B_Analog, Rxy_X, color = 'b', label = "Rxy", s = 0.1)
        #plt.scatter(B_Analog, Ryx_X, color = 'r', label = "Ryx", s = 0.1)
        #plt.scatter(B_Analog, Rho_xy, color = 'Orange', label = "Sum_Avg", s= 0.1)
        plt.scatter(B_Analog, Rho_xy_diff, color = 'black', label = "Diff_Avg", s= 0.1)
        
elif plot_Freq == True:
    if plot_V == True:
        plt.scatter(Freq, Vxy_X, color = 'b', label = "Vxy", s = 0.1)
        plt.scatter(Freq, Vyx_X, color = 'r', label = "Vyx", s = 0.1)
        plt.scatter(Freq, Rho_xy*I, color = 'Orange', label = "Ryx", s = 0.1)
    else:
        plt.scatter(Freq, Rxy_X, color = 'b', label = "Rxy", s = 0.1)
        plt.scatter(Freq, Ryx_X, color = 'r', label = "Ryx", s = 0.1)
        plt.scatter(Freq, Rho_xy, color = 'Orange', label = "Rhoxy", s= 0.1)
else:
    if plot_V == True:
        plt.scatter(Time, Vxy_X, color = 'b', label = "Vxy", s = 0.1)
        plt.scatter(Time, Vyx_X, color = 'r', label = "Vyx", s = 0.1)
        plt.scatter(Time, (Vxy_X + Vyx_X)/2, color = 'orange', s = 0.1)
    
    else:
        plt.scatter(Time, Rxy_X, color = 'b', label = "Rxy", s = 0.1)
        plt.scatter(Time, Ryx_X, color = 'r', label = "Ryx", s = 0.1)
        plt.scatter(Time, Rho_xy, color = 'Orange', label = "Rhoxy", s= 0.1)

#Noise beat of ~28s, 0.0356Hz

if plot_fit == True:
    x_ax = np.linspace(np.round(B_Analog[0]), np.round(B_Analog[-1]), len(B_Analog))
    
    #y_ax = (m * x_ax) + b
    y_ax = (m1 * (x_ax**2)) + (m2 * x_ax) + c

    plt.plot(x_ax, y_ax, color = 'purple', label = "Fit")

if timeline == True:
    #Create vertical lines to seperate 
    
    for i in slices:
        #plt.ylim(min(Rxy_X), max(Rxy_X))
        plt.plot([i, i], [timeline_ylim[0], timeline_ylim[1]], color = 'gray', alpha = 0.5)
        

if plot_B == True:
    plt.xlabel('B [T]')
elif plot_Freq == True:
    plt.xlabel('f [Hz]')
else:
    plt.xlabel('t [s]')

if plot_V == True:
    plt.ylabel('V [$V$]')
    plt.title('V, I = ' + str(I*(1E6)) +'uA')     
       
else:
    plt.ylabel('R [$\Omega$]')
    plt.title('R, I = ' + str(I*(1E6)) +'uA')  

plt.ylim([-10, 10])
    
plt.grid()    
plt.legend()
plt.show()

plt.figure()
plt.plot(B_Analog, T_Sample, 'r')
plt.xlabel("B [T]")
plt.ylabel("T [K]")

plt.figure()
plt.plot(Time, T_Sample, c = 'r')
plt.ylabel("T_Sample [K]")
plt.xlabel("t [s]")

plt.figure()
plt.plot(Time, T_VTI, c = 'r')
plt.ylabel("T_VTI [K]")
plt.xlabel("t [s]")


#%%
plt.figure()
plt.plot(Time, B_Analog)

#%%
plt.figure()
plt.plot(B_Analog, Rxy_X, c = 'b', label = "Rxy")
plt.xlabel("B [T]")
plt.ylabel("Rxy [$\Omega$]")
plt.legend()



plt.figure()
plt.plot(Time, Rxy_X, c = 'b', label = "Rxy")
plt.xlabel("t [s]")
plt.ylabel("Rxy [$\Omega$]")
plt.legend()

m_T = -7.246355e-05
m_T = -4.74E-3
Rxy_noT = Rxy_X - m_T*(T_Sample-300)
Ryx_noT = Ryx_X + m_T*(T_Sample-300)
Rho_xy_noT = (Rxy_noT + Ryx_noT)/2
Rho_xy_diff_noT = (Rxy_noT - Ryx_noT)/2

m1, m2, c = np.polyfit(B_Analog[:fit_range], Rxy_noT[:fit_range], 2)
m1, m2, c = np.polyfit(B_Analog[:fit_range], Ryx_noT[:fit_range], 2)
m1, m2, c = np.polyfit(B_Analog[:fit_range], Rho_xy_noT[:fit_range], 2)
m1, m2, c = np.polyfit(B_Analog[:fit_range], Rho_xy_diff_noT[:fit_range], 2)
#m, b = np.polyfit(B_Analog[3707:5210], Ryx_noT[3707:5210], 1)



plt.figure()
#plt.scatter(B_Analog, Rxy_noT, c = 'b', label = "R_xy (no T)", s = 0.1)
#plt.scatter(B_Analog, Ryx_noT, c = 'r', label = "R_yx (no T)", s = 0.1)
#plt.scatter(B_Analog, Rho_xy_noT, c = 'orange', label = "Rho_xy (no T)", s = 0.1)
plt.scatter(B_Analog, Rho_xy_diff_noT, c = 'black', label = "Rho_diff_xy (no T)", s = 0.1)
plt.xlabel("B [T]")
plt.ylabel('R (T Removed) [$\Omega$]')
plt.title('R fit, Temperature removed, I = ' + str(I*(1E6)) +'uA')
x_ax = np.linspace(np.round(B_Analog[0]), np.round(B_Analog[-1]), len(B_Analog))
#y_ax = (m * x_ax) + b
y_ax = (m1 * (x_ax**2)) + (m2 * x_ax) + c
plt.plot(x_ax, y_ax, color = 'purple', label = "Fit")
plt.ylim([-1,1])
plt.grid()
plt.legend()


plt.figure()
plt.plot(B_Analog, T_Sample)
'''
plt.figure()
plt.plot(B_Analog, T_Sample, c = 'r', label = "T_Sample")
plt.xlabel("B [K]")
plt.ylabel("T [K]")

m, b = np.polyfit(B_Analog[:1560], Rxy_noT[:1560], 1)

plt.figure()
plt.plot(B_Analog, Rxy_noT, c = 'teal', label = "Rxy, no Temp")
plt.plot(B_Analog, B_Analog * m + b, c = 'r', label = "fit")
plt.xlabel("B [T]")
plt.ylabel("Rxy [$\Omega$]")
plt.legend()
'''
#%%


#017, [5737]
#028, [3700]

plt.figure()
plt.plot(B_Analog[fit_range:], Rxy_X[fit_range:])

plt.figure()
plt.plot(B_Analog[fit_range:], T_Sample[fit_range:])

#plt.figure()
#plt.plot(Time, Rxy_X)

#plt.figure()
#plt.plot(Time, T_Sample)

plt.figure()
plt.plot(T_Sample[fit_range:], Rxy_X[fit_range:])

m_T, d = np.polyfit(T_Sample[fit_range:], Rxy_X[fit_range:], 1)
#m_T = -7.246355e-05


Rxy_noT = Rxy_X - m_T*(T_Sample-300)
 

plt.figure()
plt.plot(Time, Rxy_noT)

plt.figure()
plt.plot(B_Analog, Rxy_noT)

plt.figure()
plt.plot(T_Sample[fit_range:], Rxy_X[fit_range:])
plt.xlabel("Temp [K]")
plt.ylabel("Rxy [$\Omega$]")
plt.plot(T_Sample[fit_range:], T_Sample[fit_range:]*m_T + d,
         c = 'r', label = 'fit')

#plt.plot(np.linspace(290,300, len(T_Sample)), np.linspace(290,300, len(T_Sample))*m_T + d,
#         c = 'r', label = 'fit')
plt.legend()

#[1400:2000]
#m, b = np.polyfit(B_Analog[1400:2000], Rxy_X[1400:2000], 1)

#%%

y_fit =  (m * B_Analog) + b

Rxy_noHall = Rxy_X - y_fit

plt.figure()
plt.plot(B_Analog, Rxy_noHall)


#%%

m1, m2, c = np.polyfit(B_Analog, Ryx_X, 2)
#m3 = np.polyfit(B_Analog, Rxy_noT, 1)

y2_fit = (m1*(B_Analog**2) + m2*(B_Analog) + c)

plt.figure()
plt.plot(B_Analog, Ryx_X, c = 'r')
plt.plot(B_Analog, y2_fit, c = 'b')
plt.ylim([-1,1])


#%%
Rxy_noHall_no2nd = Rxy_noHall - y2_fit

plt.figure()
plt.plot(B_Analog, Rxy_noHall_no2nd)

plt.figure()
plt.plot(Time, Rxy_noHall_no2nd)

plt.figure()
plt.plot(T_Sample, Rxy_noHall_no2nd)

plt.figure()
plt.plot(B_Analog, T_Sample, c = 'r', label = "Temperature")
plt.legend()

m3, d = np.polyfit(B_Analog[160:12360], Rxy_noHall[160:12360], 1)


