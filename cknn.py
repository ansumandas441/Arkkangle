import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import classification_report
from pandas.testing import assert_frame_equal

columns=["Class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium"\
         ,"Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue"\
        ,"OD280/OD315 of diluted wines","Proline"]

df=pd.read_csv("dataset/wine.data", names=columns)

#Features 
X=df.drop("Class",axis=1).values

#Class labels
Y=df["Class"].values

#train test split 80% train 20% test 
X_train,X_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42, stratify=Y)


#main dataset
X_1=np.hstack((X_train,Y_train.reshape(len(Y_train),1)))


temp2=X_1.shape[0]


#Prototype
Z=X_1[0,:]
Z=Z.reshape(1,-1)
X_1=np.delete(X_1,0,axis=0) #deleting 0'th row

ct=0
while(1):
    
    temp=False
    i=np.random.randint(0,X_1.shape[0])
    
    dist_0=np.sqrt(np.sum((X_1[0,:]-Z[0,:])**2))
    val=0
        
    for j in range(Z.shape[0]):
            
        dist_0=np.sqrt(np.sum((X_1[i,:]-Z[0,:])**2))
        dist=np.sqrt(np.sum((X_1[i,:]-Z[j,:])**2))
        if(dist<dist_0):
            dist_0=dist
            val=j
                    
    if(int(X_1[i,0])!=int(Z[j,0])):
        Z=np.vstack((Z,X_1[i,:]))
        temp=True
    X_1=np.delete(X_1,i,axis=0) #deleting i'th row
    
    if(temp==False):
        ct+=1
    if(ct==20):    
        break
        
print("The condensed set is "+str(100*(Z.shape[0]/temp2))+"% of the main dataset\n")



#The training using condensed set

x_train=Z[:,:13]
y_train=Z[:,13]

#Setup a knn classifier with k neighbors as 9 gives a maxima in testing accuracy
knn = KNeighborsClassifier(n_neighbors=9)

#Fit the model
knn.fit(x_train,y_train)


#Get accuracy. 
print("Accuracy = "+str(100*knn.score(X_test,y_test))+" %\n")

#let predict for X_test using the classifier we had fit above
y_pred = knn.predict(X_test)

print(classification_report(y_test,y_pred))








