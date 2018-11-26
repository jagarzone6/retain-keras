#!/usr/bin/env python
# coding: utf-8

# In[317]:


import numpy as np
import pandas as pd
import random
from random import randint
from random import uniform
from sklearn.model_selection import train_test_split
import pickle


numeric_size = 3
num_of_patients = 15000

def get_rand_visit_codes():
    #num_of_codes_in_visit = randint(1,5)
    #visit_c = random.sample(range(0,68), num_of_codes_in_visit)
    num_of_codes_in_visit = 2
    visit_c = []
    #print (visit_c)
    return visit_c
ite_rand = 0
def get_rand_visit_numerics():
    #[Diastolic,Systolic, pulse]
    global ite_rand
    if ite_rand == 0:
        visit_n = [uniform(60,90),uniform(90,160),uniform(60,100)]
    else:
        visit_n = [uniform(90,110),uniform(160,180),uniform(100,120)]
    #print (visit_n)
    return visit_n

def create_patient():
    num_of_visits = randint(5,20)
    global ite_rand
    if ite_rand == 0:
        ite_rand = 1
    else:
        ite_rand = 0
    anomaly_patient_count = 0
    patient_c = []
    for k in range(0,num_of_visits,1):
        patient_c.append(get_rand_visit_codes())
    #print (patient_c)

    patient_n = []
    for k in range(0,num_of_visits,1):
        visit_numerics = get_rand_visit_numerics()
        patient_n.append(visit_numerics)
        anomaly_patient_count = anomaly_patient_count + validate_numerics_for_anomaly(visit_numerics)
        
    #print ("New patient "+str(len(target))+", # of visits: "+str(num_of_visits))
    #print ("Number of anomalies: "+ str(anomaly_patient_count))
    if anomaly_patient_count/num_of_visits > 0.7:
        target.append(1)
        #print ("Target: "+ str(1))
    else:
        target.append(0)
        #print ("Target: "+ str(0))
    #print("Target value in list: "+str(target[len(target)-1]))
    return [patient_c,patient_n,None,anomaly_patient_count]

def validate_numerics_for_anomaly(numerics):
    #[Diastolic,Systolic, pulse]
    if (numerics[0] >= 100 or numerics[0] <= 55):
        return 1
    elif (numerics[1] >= 170 or numerics[1] <= 85):
        return 1
    elif (numerics[2] >= 110 or numerics[2] <= 55):
        return 1
    else:
        return 0


# In[318]:


target=[]
df = pd.DataFrame([create_patient()], columns=['codes','numerics','to_event','anomalies'])


# In[319]:


k=1
for k in range(1,num_of_patients,1):
    if (k == 1000 or k == 3000 or k == 5000 or k == 8000 or k == 10000 or k == 15000 or k == 25000 or k == 35000):
        print ("created_patient: "+str(k))
    df.loc[k] = create_patient()
#print(target)
print(str(target.count(1)))
print(str(target.count(0)))


# In[321]:


sort_indicies = np.argsort(list(map(len, df['codes'].tolist())))
sort_indicies


# In[322]:


all_data =df.iloc[sort_indicies].reset_index()


# In[325]:


tg = pd.DataFrame([target[0]], columns=['target'])
k=1
for k in range(1,num_of_patients,1):
    tg.loc[k] = target[k]


# In[326]:


all_targets = tg.iloc[sort_indicies].reset_index()


# In[328]:


train_proportion=0.7
data_train,data_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
target_train,target_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)


# In[329]:


out_directory= 'data'


# In[330]:


data_train.sort_index().to_pickle(out_directory+'/data_train.pkl')
data_test.sort_index().to_pickle(out_directory+'/data_test.pkl')


# In[331]:


target_train.sort_index().to_pickle(out_directory+'/target_train.pkl')
target_test.sort_index().to_pickle(out_directory+'/target_test.pkl')


# In[332]:


from icd9 import ICD9
tree = ICD9('codes_pretty_printed.json')


# In[333]:


labels_icd9 = []
ids = []


for k in range(390,459,1):
    ids.append(k-390)
    if tree.find(str(k)) is not None:
        #print(tree.find(str(k)).description)
        labels_icd9.append(tree.find(str(k)).description)
    else:
        #print(tree.find('390-459').description+str(k))
        labels_icd9.append(tree.find('390-459').description+str(k))
        
#print(labels_icd9)        
#print(ids)        
ids.append(70)
ids.append(71)
ids.append(72)

labels_icd9.append('Diastolic')
labels_icd9.append('Systolic')
labels_icd9.append('Pulse')

types = dict(zip(ids,labels_icd9))


# In[334]:


#print(types)


# In[335]:


pickle.dump(types, open(out_directory+'/dictionary.pkl', 'wb'), -1)


# In[336]:


data_test.sort_index()


# In[337]:


target_test.sort_index()


# In[338]:


#data_test.sort_index().loc[13]


# In[ ]:



