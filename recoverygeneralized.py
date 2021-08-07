import random
import csv
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import struct
import maingeneralized


Name = []
Glucose = []
BloodPressure = []
BMI = []
Age = []
Desease = []

print(unseed)
print(importdataset)


with open("preserved.csv","r") as csv_file:
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
         

preserved = [Glucose, BloodPressure, BMI, Age]

print("preserved dataset: \n\n"+str(preserved))
recoveryDataset = []

for aa in range(4):
  Dataset = preserved[aa]
  D = len(Dataset) 

  recoveryColumn = []

  n = math.floor((abs(D)/g)) - 1
  l = 1
  dcount = 0
  wcount = 0
  
  
  
  
  for x in range(n+1):
   
     
      qDoubleDash=[]
      watermark=[]
      for i in range(g):
          qDoubleDash.append(Dataset[dcount+i])
      print(qDoubleDash)
                  
      print("[+] Printing the original array and seeds\n")
      print("[>] Array : ", qDoubleDash)
      print("[>] unseed : ", unseed)
      # Mapping the seeds with array using dictionary
      array_with_corresponding_unseed = []
      for i in range(0, len(unseed)):
          array_with_corresponding_unseed.append([unseed[i], qDoubleDash[i]])
           
      # using sorted() and index()
      sorted_array_with_corresponding_unseed = sorted(array_with_corresponding_unseed)
      print(sorted_array_with_corresponding_unseed)
      # Storing the result
      sorted_unseed = []
      unseeded_qDoubleDash = []
      for i in sorted_array_with_corresponding_unseed:
          sorted_unseed.append(i[0])
          unseeded_qDoubleDash.append(i[1])
      # Printing the result
      print("[+] Printing the sorted array and seeds\n")
      print("[>] Array : ", unseeded_qDoubleDash)
      print("[>] unseeds : ", sorted_unseed)



     
      
      watermark=[]
      for i in range(g-1):
          watermark.append([full_watermark[wcount]])
      print(watermark)
      
      
      qCurl1=0
      for i in range(g):
          print(weights[i])
          print(unseeded_qDoubleDash[i])
          qCurl1=qCurl1+(weights[i]*unseeded_qDoubleDash[i])
          print(qCurl1)
      print(qCurl1)
      qCurl1=math.floor(qCurl1/total_weight)
      print(qCurl1)
      
      
      qCurl=[qCurl1]
      for i in range(1,g):
          qCurl.append(unseeded_qDoubleDash[i]-unseeded_qDoubleDash[0])
          #if qDash[i]<0:
              #signed bit into unsigned bit
           #   qDash[i]=qDash[i]+2**32
      print(qCurl)
      
      qDash=[qCurl[0]]
      for i in range(1,g):
          qDash.append(math.floor(qCurl[i]/2))
          #qDash[i]=qDash[i]-2**32
          
      print(qDash)  
      
      for i in range(g):
          if qDash[i]>1000:
              qDash[i]=qDash[i]-2**32
              print(qDash[i])
      print(qDash)
      
      glist1=0
      for i in range(1,g):
          glist1=glist1+(weights[i]*qDash[i])
      print(glist1)
      glist1=math.floor(glist1/total_weight)

      glist1=qDash[0]-glist1
      print(glist1)
          
      glist=[glist1]
      for i in range(1,g):
          glist.append(qDash[i]+glist1)
          
      print(glist)
      
       
      for i in range(g):
          recoveryColumn.append(glist[i])
      
      print(recoveryColumn)
      
      dcount += g
      wcount += g-1
      
  recoveryDataset.append(recoveryColumn)



print("\n\n\n\nRecovery dataset: \n\n"+str(recoveryDataset))

rows = zip(Name, recoveryDataset[0], recoveryDataset[1], recoveryDataset[2], recoveryDataset[3], Desease)

with open("new_recovery.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Desease'])
    for row in rows:
        writer.writerow(row)



    
       
    
    

 





