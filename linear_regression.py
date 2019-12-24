import pandas  as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


#LINEAR REGRESSION
def initialize_parameters(lenw, b=0):
    w= np.random.randn(1,lenw)
    return w,b
def forward_prop(X,w,b):
    z=np.dot(w,X) +b #hypothesis
    return z
def cost_function(z,y):
    m= y.shape[0]
    J=(1/(2*m))*np.sum(np.square(z-y)) #cost function
    return J


def back_prop(X,y,z):
    m=y.shape[0]
    dz=(1/m)*(z-y)
    dw=np.dot(dz,X.T)
    db=np.sum(dz)
    return dw,db

def gradient_descent_update(w,b,dw,db,learning_rate=10):
   w= w-learning_rate*dw
   b= b-learning_rate*db
   return w,b
def linear_regression_model(x_train,y_train,x_val,y_val,epochs,learning_rate=10):
   lenw=x_train.shape[0]
   lenw2=x_val.shape[0]
   w,b=initialize_parameters(lenw)
   costs_train =[]
   m_train = y_train.shape[0]
   m_val=y_val.shape[0]
   for i in range(1,epochs+1):
       z_train= forward_prop(x_train,w,b)
       cost_train=cost_function(z_train,y_train)
       dw,db=back_prop(x_train,y_train,z_train)
       w,b=gradient_descent_update(w,b,dw,db,learning_rate)
       if i%10==0:
           costs_train.append(cost_train)
       #MAE_train
       MAE_train=(1/m_train)*np.sum(np.abs(z_train-y_train))

       #cost_val, MAE_VAL
       w2,b2=initialize_parameters(lenw2)
       z_val =forward_prop(x_val,w2,b2)
       cost_val = cost_function(z_val,y_val)
       MAE_val=(1/m_val)*np.sum(np.abs(z_val-y_val))

       print('Epochs '+str(i)+'/'+str(epochs)+':')
       print('Training cost '+str(cost_train)+'|'+'Validation cost '+str(cost_val))
       print('Training MAE '+str(MAE_train)+'|'+'Validation MAE'+str(MAE_val))
   plt.plot(costs_train)
   plt.xlabel('Iterations(per tens)')
   plt.ylabel('Training Cost') 
   plt.title('Learning rate'+str(learning_rate))
   plt.show()
  






   
df= pd.read_csv("Salary_Data.csv")

x_train=df['YearsExperience']
y_train=df['Salary']
plt.plot(x_train,y_train)
vf=df.iloc[:10]
x_val=vf['YearsExperience']
y_val=vf['Salary']
linear_regression_model(x_train,y_train,x_val,y_val,2,0.01)
