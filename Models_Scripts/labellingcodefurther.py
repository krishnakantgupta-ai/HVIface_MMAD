Human_loc=[87,88,89,90,92,93,94,95,96,113,114,129,131,132]
value=[[75,76,74],[75],[78,121,122],[121,122,94],[117,95,122,91,94,90],[90,87,86],[86,90],[86],[81,124],[79],[81,79,80,122],[78,79],[76,77,79],[75,76,78,129,121]]


lst= [j-44 for i in value for j in i]

from itertools import islice

def convert(lst, var_lst):
    idx = 0
    for var_len in var_lst:
        yield lst[idx : idx + var_len]
        idx += var_len


var_lst = [3,1,3,3,6,3,2,1,2,1,4,2,3,5]
lst1=list(convert(lst, var_lst))
print(lst1)

test1={}
for i in range(len(Human_loc)):
    for j in range(len(lst1[i])):
       
        test1.update({Human_loc[i]-77:lst1[i]})
   


print(test1)


Li1=[]
len_vir=166


for i in test1:
    #print(len(test1[i]))
    for j in range(len(test1[i])):
        Li1.append((i-1)*len_vir+test1[i][j])
       
print(Li1)

Li1.sort()
print(len(Li1))



import pandas as pd


df = pd.read_csv("3REA_test_df.csv")
b=df.iloc[:,[0]]
print(len(b))
j=0
for i in range(len(b)):
    for j in range(len(Li1)):
        if Li1[j]==df.iloc[i,0]:
            df.iloc[i,1]=1
            break
'''       
for i in range(len(df)):
    if df.loc[i,'rsa.m$value']==0 and df.loc[i,'inf.m$value']==1:
        df.loc[i,'inf.m$value']=0   
'''        
       
df.to_csv("3REA_test_df.csv")