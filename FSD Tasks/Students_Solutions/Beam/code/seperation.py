import numpy as np
import pandas as pd
import pickle
import math


train_labels = df.label.to_numpy()
train_data = np.stack(df.sensor_data.to_numpy())
df_np = df.to_numpy()
#df is the train dataframe, df_np the corresponding numpy array

df_test_np = df_test.to_numpy()
#df_test is the test dataframe, df_test_np the corresponding numpy array
print(df_test_np.shape)
print(df_np.shape)
#print(df_test_np[200:500,4])
#Investigation how many distinct models occur in the train/test data and the number of samples of each model

for j in range (1001):
    if np.isnan(df_test_np[j,4]) == True:
        df_test_np[j,4] = 0
print(df_test_np[400:500,4])
#transform nan entries into 0 in the array

k = 1
counter = np.zeros(shape = (100), dtype= object)
for i in range (1, 1001):
    
    counter[0] = math.floor(df_test_np[0,4]/100)*100
    counter[k] = math.floor(df_test_np[i,4]/100)*100
    #we split the data in set of hundreds, so for exapmle the mass 1535 is in the group 1500
    proof = 0
    for l in range (k):
        
        if counter[k] == counter[l]:
            proof += 1
    if proof == 0:
        k +=1
    else: 
        counter[k] = 0
print(counter)



#for example here we count how many models E_(Rifter_Partner) are in the test set (the range is 1001)

#Seperation of the train data by model, there are 363x Variant Comfortline, 1180x Passat 3C and 2510x Golf VI
Comfortline_data = np.zeros(shape = (363,6), dtype = object)
Passat_data = np.zeros(shape=(1180,6), dtype = object)
Golf_data = np.zeros(shape = (2510, 6), dtype = object)
i = 0
j = 0
l = 0
for k in range (4053):
    if df_np[k,2] == 'Golf VI':
        Golf_data[i,:] = df_np[k,:]
        i += 1
    elif df_np[k,2] == 'Variant Comfortline':
        Comfortline_data[j,:] = df_np[k,:]
        j += 1
    elif df_np[k,2] == 'Passat 3C':
        Passat_data[l,:] = df_np[k,:]
        l += 1
      
#convert back in df-framework, each dataframe consits only of data with the particular model in the name
Comfortline_df = pd.DataFrame(Comfortline_data)
Passat_df = pd.DataFrame(Passat_data)
Golf_df = pd.DataFrame(Golf_data)

#Seperation of the test data by model, there are 358x Variant Comfortline, 456x forfour, 
#25x GY_A3, NX4E_Tucson, Daimler 212 and Daimler F2B, 23x Daimler 906, OS_OSE_Kona and E_Rifter_Partner
#and 18x the CV_EV6
Comfortline_data_test = np.zeros(shape = (358,6), dtype = object)
forfour_data_test = np.zeros(shape=(456,6), dtype = object)
GY_A3_data_test = np.zeros(shape = (25, 6), dtype = object)
NX4E_Tucson_data_test = np.zeros(shape = (25, 6), dtype = object)
OS_OSE_Kona_FL1_data_test= np.zeros(shape = (23, 6), dtype = object)
CV_EV6_data_test = np.zeros(shape = (18, 6), dtype = object)
Daimler_906_data_test = np.zeros(shape = (23, 6), dtype = object)
Daimler_F2B_data_test = np.zeros(shape = (25, 6), dtype = object)
Daimler_212_data_test = np.zeros(shape = (25, 6), dtype = object)
E_Rifter_Partner_data_test = np.zeros(shape = (23, 6), dtype = object) 

b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0 
i = 0
j = 0
l = 0

for k in range (1001):
    if df_test_np[k,2] == 'Variant Comfortline':
        Comfortline_data_test[b,:] = df_test_np[k,:]
        b += 1
    elif df_test_np[k,2] == 'forfour':
        forfour_data_test[c,:] = df_test_np[k,:]
        c += 1
    elif df_test_np[k,2] == 'GY_(A3)':
        GY_A3_data_test[d,:] = df_test_np[k,:]
        d += 1
    elif df_test_np[k,2] == 'NX4E_(Tucson)':
        NX4E_Tucson_data_test[e,:] = df_test_np[k,:]
        e += 1
    elif df_test_np[k,2] == 'OS_OSE_(Kona_FL1)':
        OS_OSE_Kona_FL1_data_test[f,:] = df_test_np[k,:]
        f += 1
    elif df_test_np[k,2] == 'CV_(EV6)':
        CV_EV6_data_test[g,:] = df_test_np[k,:]
        g += 1
    elif df_test_np[k,2] == 'Daimler_906_FL3A4_KL3A4_KL3A5_(Mercedes_Sprinter_MJ18)':
        Daimler_906_data_test[h,:] = df_test_np[k,:]
        h += 1
    elif df_test_np[k,2] == 'Daimler_F2B_(GLA-Klasse)':
        Daimler_F2B_data_test[i,:] = df_test_np[k,:]
        i += 1
    elif df_test_np[k,2] == 'Daimler_212_R1ES_R1EC_(E-Klasse_BM213_FL1)':
        Daimler_212_data_test[j,:] = df_test_np[k,:]
        j += 1
    elif df_test_np[k,2] == 'E_(Rifter_Partner)':
        E_Rifter_Partner_data_test[l,:] = df_test_np[k,:]
        l += 1

#convert back in df-framework
Comfortline_test_df = pd.DataFrame(Comfortline_data_test)
forfour_test_df = pd.DataFrame(forfour_data_test)
GY_A3_test_df = pd.DataFrame(GY_A3_data_test)
NX4E_Tucson_test_df = pd.DataFrame(NX4E_Tucson_data_test)
OS_OSE_Kona_FL1_test_df = pd.DataFrame(OS_OSE_Kona_FL1_data_test)
CV_EV6_test_df = pd.DataFrame(CV_EV6_data_test)
Daimler_906_test_df = pd.DataFrame(Daimler_906_data_test)
Daimler_F2B_test_df = pd.DataFrame(Daimler_F2B_data_test)
Daimler_212_test_df = pd.DataFrame(Daimler_212_data_test)
GE_Rifter_Partner_test_df = pd.DataFrame(E_Rifter_Partner_data_test)

##############################################################################################################################################################
##############################################################################################################################################################
#Seperation by mass for train data

mass1341_data = np.zeros(shape = (2510,6), dtype = object)
mass1594_data = np.zeros(shape=(1180,6), dtype = object)
mass1530_data = np.zeros(shape = (363, 6), dtype = object)

i = 0
j = 0
l = 0
for k in range (4053):
    if df_np[k,4] == 1341:
        mass1341_data[i,:] = df_np[k,:]
        i += 1
    elif df_np[k,4] == 1594:
        mass1594_data[j,:] = df_np[k,:]
        j += 1
    elif df_np[k,4] == 1530:
        mass1530_data[l,:] = df_np[k,:]
        l += 1

mass1341_data_df = pd.DataFrame(mass1341_data)
mass1530_data_df = pd.DataFrame(mass1530_data)
mass1594_data_df = pd.DataFrame(mass1594_data)

#seperation by mass for test data

z = 0
for i in range (1001):
    if math.floor(df_test_np[i,4]/100)*100 == 3500:
        z +=1
print(z)

mass1500_data_test = np.zeros(shape = (383,6), dtype = object)
massNaN_data_test = np.zeros(shape = (456,6), dtype = object)
mass1600_data_test = np.zeros(shape=(35,6), dtype = object)
mass1200_data_test = np.zeros(shape = (5, 6), dtype = object)
mass1400_data_test = np.zeros(shape = (8, 6), dtype = object)
mass1900_data_test = np.zeros(shape = (8, 6), dtype = object)
mass1800_data_test = np.zeros(shape = (20, 6), dtype = object)
mass1700_data_test = np.zeros(shape = (25, 6), dtype = object)
mass2000_data_test = np.zeros(shape = (19, 6), dtype = object)
mass2200_data_test = np.zeros(shape = (8, 6), dtype = object)
mass2100_data_test = np.zeros(shape = (11, 6), dtype = object)
mass2900_data_test = np.zeros(shape = (5, 6), dtype = object)
mass3100_data_test = np.zeros(shape = (5, 6), dtype = object)
mass3200_data_test = np.zeros(shape = (5, 6), dtype = object)
mass2800_data_test = np.zeros(shape = (5, 6), dtype = object)
mass3500_data_test = np.zeros(shape = (3, 6), dtype = object)




a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0 
i = 0
j = 0
l = 0 
m = 0
n= 0
o = 0
r = 0
p = 0

for k in range (1001):
    if math.floor(df_test_np[k,4]/100)*100 == 1500:
        mass1500_data_test[b,:] = df_test_np[k,:]
        b += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 0:
        massNaN_data_test[c,:] = df_test_np[k,:]
        c += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 1600:
        mass1600_data_test[d,:] = df_test_np[k,:]
        d += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 1200:
        mass1200_data_test[e,:] = df_test_np[k,:]
        e += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 1400:
        mass1400_data_test[f,:] = df_test_np[k,:]
        f += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 1900:
        mass1900_data_test[g,:] = df_test_np[k,:]
        g += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 1800:
        mass1800_data_test[a,:] = df_test_np[k,:]
        a += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 1700:
        mass1700_data_test[h,:] = df_test_np[k,:]
        h += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 2000:
        mass2000_data_test[i,:] = df_test_np[k,:]
        i += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 2200:
        mass2200_data_test[j,:] = df_test_np[k,:]
        j += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 2100:
        mass2100_data_test[l,:] = df_test_np[k,:]
        l += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 2900:
        mass2900_data_test[m,:] = df_test_np[k,:]
        m += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 3100:
        mass3100_data_test[n,:] = df_test_np[k,:]
        n += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 3200:
        mass3200_data_test[o,:] = df_test_np[k,:]
        o += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 2800:
        mass2800_data_test[p,:] = df_test_np[k,:]
        p += 1
    elif math.floor(df_test_np[k,4]/100)*100 == 3500:
        mass3500_data_test[r,:] = df_test_np[k,:]
        r += 1

print(mass2200_data_test[:,4])

#convert back in df-framework

mass1500_data_test = pd.DataFrame(mass1500_data_test)
massNaN_data_test = pd.DataFrame(massNaN_data_test)
mass1600_data_test = pd.DataFrame(mass1600_data_test)
mass1200_data_test = pd.DataFrame(mass1200_data_test)
mass1400_data_test = pd.DataFrame(mass1400_data_test)
mass1900_data_test = pd.DataFrame(mass1900_data_test)
mass1800_data_test = pd.DataFrame(mass1800_data_test)
mass1700_data_test = pd.DataFrame(mass1700_data_test)
mass2000_data_test = pd.DataFrame(mass2000_data_test)
mass2200_data_test = pd.DataFrame(mass2200_data_test)
mass2100_data_test = pd.DataFrame(mass2100_data_test)
mass2900_data_test = pd.DataFrame(mass2900_data_test)
mass3100_data_test = pd.DataFrame(mass3100_data_test)
mass3200_data_test = pd.DataFrame(mass3200_data_test)
mass2800_data_test = pd.DataFrame(mass2800_data_test)
mass3500_data_test = pd.DataFrame(mass3500_data_test)


##############################################################################################################################################################
##############################################################################################################################################################
#Seperation by velocity for train data

kmh21_data = np.zeros(shape = (830,6), dtype = object)
kmh20_data = np.zeros(shape=(2355,6), dtype = object)
kmh19_data = np.zeros(shape = (641, 6), dtype = object)
kmh0_data = np.zeros(shape = (16, 6), dtype = object)
kmh22_data= np.zeros(shape = (125, 6), dtype = object)
kmh23_data = np.zeros(shape = (11, 6), dtype = object)
kmh18_data = np.zeros(shape = (58, 6), dtype = object)
kmh13_data = np.zeros(shape = (1, 6), dtype = object)
kmh15_data = np.zeros(shape = (2, 6), dtype = object)
kmh24_data = np.zeros(shape = (4, 6), dtype = object)
kmh17_data = np.zeros(shape = (8, 6), dtype = object)
kmh16_data = np.zeros(shape = (2, 6), dtype = object)

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0 
i = 0
j = 0
l = 0 
m = 0

for k in range (4053):
    if df_np[k,3] == 21:
        kmh21_data[b,:] = df_np[k,:]
        b += 1
    elif df_np[k,3] == 20:
        kmh20_data[c,:] = df_np[k,:]
        c += 1
    elif df_np[k,3] == 19:
        kmh19_data[d,:] = df_np[k,:]
        d += 1
    elif df_np[k,3] == 0:
        kmh0_data[e,:] = df_np[k,:]
        e += 1
    elif df_np[k,3] == 22:
        kmh22_data[f,:] = df_np[k,:]
        f += 1
    elif df_np[k,3] == 23:
        kmh23_data[g,:] = df_np[k,:]
        g += 1
    elif df_np[k,3] == 18:
        kmh18_data[h,:] = df_np[k,:]
        h += 1
    elif df_np[k,3] == 13:
        kmh13_data[i,:] = df_np[k,:]
        i += 1
    elif df_np[k,3] == 15:
        kmh15_data[j,:] = df_np[k,:]
        j += 1
    elif df_np[k,3] == 24:
        kmh24_data[l,:] = df_np[k,:]
        l += 1
    elif df_np[k,3] == 17:
        kmh17_data[m,:] = df_np[k,:]
        m += 1
    elif df_np[k,3] == 16:
        kmh16_data[a,:] = df_np[k,:]
        a += 1

#convert back in df-framework
kmh21_data_df = pd.DataFrame(kmh21_data)
kmh20_data_df = pd.DataFrame(kmh20_data)
kmh19_data_df = pd.DataFrame(kmh19_data)
kmh0_data_df = pd.DataFrame(kmh0_data)
kmh22_data_df = pd.DataFrame(kmh22_data)
kmh23_data_df = pd.DataFrame(kmh23_data)
kmh18_data_df = pd.DataFrame(kmh18_data)
kmh13_data_df = pd.DataFrame(kmh13_data)
kmh15_data_df = pd.DataFrame(kmh15_data)
kmh24_data_df = pd.DataFrame(kmh24_data)
kmh17_data_df = pd.DataFrame(kmh17_data)
kmh16_data_df = pd.DataFrame(kmh16_data)


#for test data:

kmh30_data_test = np.zeros(shape = (358,6), dtype = object)
kmh35_data_test = np.zeros(shape=(456,6), dtype = object)
kmh20_data_test = np.zeros(shape = (187, 6), dtype = object)
b = 0
c = 0
d = 0

for k in range (1001):
    if df_test_np[k,3] == 30:
        kmh30_data_test[b,:] = df_test_np[k,:]
        b += 1
    elif df_test_np[k,3] == 35:
        kmh35_data_test[c,:] = df_test_np[k,:]
        c += 1
    elif df_test_np[k,3] == 20:
        kmh20_data_test[d,:] = df_test_np[k,:]
        d += 1

#convert back in df-framework
kmh30_data_test_df = pd.DataFrame(kmh30_data_test)
kmh20_data_test_df = pd.DataFrame(kmh20_data_test)
kmh35_data_test_df = pd.DataFrame(kmh35_data_test)

##############################################################################################################################################################
##############################################################################################################################################################

#Seperation by deceleration average for the train data
#using the round-function to get integers with distinguishable deceleration averages

decel3_data = np.zeros(shape = (46,6), dtype = object)
decel4_data = np.zeros(shape=(739,6), dtype = object)
decel5_data = np.zeros(shape = (2579, 6), dtype = object)
decel6_data = np.zeros(shape = (673, 6), dtype = object)
decel7_data = np.zeros(shape = (15, 6), dtype = object)

b = 0
c = 0
d = 0
e = 0
f = 0

for k in range (4053):
    if round(df_np[k,5]) == -3:
        decel3_data[b,:] = df_np[k,:]
        b += 1
    elif round(df_np[k,5]) == -4:
        decel4_data[c,:] = df_np[k,:]
        c += 1
    elif round(df_np[k,5]) == -5:
        decel5_data[d,:] = df_np[k,:]
        d += 1
    elif round(df_np[k,5]) == -6:
        decel6_data[e,:] = df_np[k,:]
        e += 1
    elif round(df_np[k,5]) == -7:
        decel7_data[f,:] = df_np[k,:]
        f += 1

#convert back in df-framework
decel3_data_df = pd.DataFrame(decel3_data)
decel4_data_df = pd.DataFrame(decel4_data)
decel5_data_df = pd.DataFrame(decel5_data)
decel6_data_df = pd.DataFrame(decel6_data)
decel7_data_df = pd.DataFrame(decel7_data)

#Seperation by deceleration average for the test data
#using the round-function to get integers to get distinguishable deceleration averages


decel2_data_test = np.zeros(shape = (7,6), dtype = object)
decel3_data_test = np.zeros(shape = (219,6), dtype = object)
decel4_data_test = np.zeros(shape=(298,6), dtype = object)
decel5_data_test = np.zeros(shape = (193, 6), dtype = object)
decel6_data_test = np.zeros(shape = (181, 6), dtype = object)
decel7_data_test = np.zeros(shape = (103, 6), dtype = object)

b = 0
c = 0
d = 0
e = 0
f = 0
g = 0

for k in range (1001):
    if round(df_test_np[k,5]) == -2:
        decel2_data_test[b,:] = df_test_np[k,:]
        b += 1
    elif round(df_test_np[k,5]) == -3:
        decel3_data_test[c,:] = df_test_np[k,:]
        c += 1
    elif round(df_test_np[k,5]) == -4:
        decel4_data_test[d,:] = df_test_np[k,:]
        d += 1
    elif round(df_test_np[k,5]) == -5:
        decel5_data_test[e,:] = df_test_np[k,:]
        e += 1
    elif round(df_test_np[k,5]) == -6:
        decel6_data_test[f,:] = df_test_np[k,:]
        f += 1
    elif round(df_test_np[k,5]) == -7:
        decel7_data_test[g,:] = df_test_np[k,:]
        g += 1

#convert back in df-framework
decel2_data_test_df = pd.DataFrame(decel2_data_test)
decel3_data_test_df = pd.DataFrame(decel3_data_test)
decel4_data_test_df = pd.DataFrame(decel4_data_test)
decel5_data_test_df = pd.DataFrame(decel5_data_test)
decel6_data_test_df = pd.DataFrame(decel6_data_test)
decel7_data_test_df = pd.DataFrame(decel7_data_test)