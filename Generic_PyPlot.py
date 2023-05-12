import csv
import matplotlib.pyplot as plt
import pandas as pd

    
#%%
# file  = '230510_GaAs_D181211Ai_1V_1Mohm_LIA8_12_Rxx_Ryy_4K_p1T_m1T_Bsweep.csv'
file = '230510_GaAs_D181211Ai_1uA_LIA8_12_Rxy_Ryx_4K_m8T_p8T_Bsweep.csv'
data = pd.read_csv(file)

# print(data)

# plt.scatter(data.An_Field, data.Vxx_x, color = 'g')
# plt.scatter(data.An_Field, data.Vyy_x, color = 'b')

plt.plot(data.An_Field, data.Vxy_x, color = 'g')
plt.plot(data.An_Field, data.Vyx_x, color = 'b')

plt.xlabel('B')
plt.ylabel('Vxy_x')
plt.title('Title')
plt.grid()
#plt.legend()
plt.show()