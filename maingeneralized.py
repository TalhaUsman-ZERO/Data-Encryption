import random
import csv
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import numpy


Name = []
Glucose = []
BloodPressure = []
BMI = []
Age = []
Desease = []

#generate random group size 
g=(random.randint(2,10))
print(g)

#generate weights
#weights range [0, g-1]
from numpy import random
weights=random.randint(1,g,size=(g),dtype='int64')
print(weights)


#total weighted sum
total_weight=0
for i in range(g):
    total_weight=total_weight+weights[i]
print(total_weight)
    

#import dataset according to the generated group size
importdataset=math.floor(156/g)
print(importdataset)

#calculating watermark_size
watermark_size=156-importdataset
print(watermark_size)

from numpy import random
full_watermark=random.randint(2,size=(watermark_size+1))
print(full_watermark)


importdataset=importdataset*g
print(importdataset)


#generate seed of distint random numbers
import random
Seed=random.sample(range(1, 11), g)
print(Seed)

unseed=[]



with open("Diabetes Preprocessed Random.csv","r") as csv_file:
    csv_reader=csv.DictReader(csv_file,delimiter=',')
    csv_reader=[row for i, row in enumerate(csv_reader) if i in range(importdataset)]
    print(csv_reader)
    
    for lines in csv_reader:
      Name.append(lines['Name'])
      Glucose.append(int(float(lines['Glucose'])))
      BloodPressure.append(int(float(lines['BloodPressure'])))
      BMI.append(round(float(lines['BMI'])))
      Age.append(int(lines['Age']))
      Desease.append(lines['Desease'])
         


      
# unset = 0 ---- nth bit starts form right ---- x number
def unset_nth_bit(x: int, n: int):
  return bin(x & ~(1 << n))

# set = 1 ---- nth bit.....starts form right ---- x number
def set_nth_bit(x: int, n: int):
  return bin(x | 1 << n)
      
      
unpreserved = [Glucose, BloodPressure, BMI, Age]

print("unpreserved dataset: \n\n"+str(unpreserved))
preservedDataset = []


for aa in range(4):
  Dataset = unpreserved[aa]
  D = len(Dataset) #156

  preservedColumn = []
  
  n = math.floor((abs(D)/g)) - 1
  print(n)
  l = 1
  dcount = 0
  wcount = 0
  #first_column = 0; #add watermark if first
  
  for x in range(n+1):
      glist=[]
      watermark=[]
      for i in range(g):
          glist.append(Dataset[dcount+i])
      print(glist)
      
      for i in range(g-1):
          watermark.append(full_watermark[wcount+i])
          print(watermark)
          
      qDash1=0
      for i in range(g):
          qDash1=qDash1+(weights[i]*glist[i])
      print(qDash1)
      qDash1=math.floor(qDash1/total_weight)
      print(qDash1)
          
      qDash=[qDash1]
      for i in range(1,g):
          qDash.append(glist[i]-glist[0])
          print(qDash)
          if qDash[i]<0:
              #signed bit into unsigned bit
              qDash[i]=qDash[i]+2**32
      print(qDash)
          
          
      qCurl=[qDash[0]]
      for i in range(1,g):
          qCurl.append(qDash[i]*2)
          #if qCurl[i]<0:
           #   qCurl[i]=qCurl[i]+2**32
          
      print(qCurl)    
              
      binary=[] 
      for i in range(g):
          binary.append(bin(qCurl[i]))
      print(binary)
      
      for y in range(g-1):
          if (watermark[y] == 0):
              binary[y+1] = unset_nth_bit(int(binary[y+1], 2), 0)
          elif (watermark[y] == 1):
              binary[y+1] = set_nth_bit(int(binary[y+1], 2),0)
        
      print(binary) 
      
      qDoubleDash1=0
      for i in range(1,g):
          print(weights[i],binary[i][2:],2)
          qDoubleDash1=qDoubleDash1+(weights[i]*int(binary[i][2:],2))
      print(qDoubleDash1)
      qDoubleDash1=math.floor(qDoubleDash1/total_weight)
      print(qDoubleDash1)
      
      print(qCurl[0])

      qDoubleDash1=qCurl[0]-qDoubleDash1
      print(qDoubleDash1)
     
      #qDoubleDash1 = qCurl[0] - math.floor((weights[1]*int(binary[1][2:], 2) + weights[2]*int(binary[2][2:], 2) + weights[3]*int(binary[3][2:], 2)) / (total_weight))

  
      qDoubleDash=[qDoubleDash1]
      print(qDoubleDash)
      for i in range(1,g):
          qDoubleDash.append(int(binary[i][2:],2)+qDoubleDash1)
          
      print(qDoubleDash)
      
          # printing original list
      print("[+] Printing the original array and seeds\n")
      print("[>] Array : ", qDoubleDash)
      print("[>] Seeds : ", Seed)
      # Mapping the seeds with array using dictionary
      array_with_corresponding_seed = []
            
      for i in range(0, len(Seed)):
          array_with_corresponding_seed.append([Seed[i], qDoubleDash[i]])
       
      print(array_with_corresponding_seed)
            
      temp=[]
      for j in range(0,11):
          for i in array_with_corresponding_seed:
              if j==i[0]:
                  temp.append(array_with_corresponding_seed.index(i))
      # using sorted() and index()
      sorted_array_with_corresponding_seed = sorted(array_with_corresponding_seed)
    
      # Storing the result
      sorted_Seed = []
      seeded_qDoubleDash = []
      for i in sorted_array_with_corresponding_seed:
          sorted_Seed.append(i[0])
          seeded_qDoubleDash.append(i[1])
      # Printing the result
      print("[+] Printing the sorted array and seeds\n")
      print("[>] Array : ", seeded_qDoubleDash)
      print("[>] Seeds : ", sorted_Seed)
      print("[>]temp: ",temp)
      unseed=temp.copy()

      
      
   
      
     
      for i in range(g):
          preservedColumn.append(seeded_qDoubleDash[i])
          
      print("preserved columns")
      print(preservedColumn)
      
      dcount += g
      wcount += g-1
     
  preservedDataset.append(preservedColumn)
  
  
print("\n\n\n\npreserved dataset: \n\n"+str(preservedDataset))

rows = zip(Name, preservedDataset[0], preservedDataset[1], preservedDataset[2], preservedDataset[3], Desease)

with open("preserved.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Desease'])
    for row in rows:
        writer.writerow(row)


