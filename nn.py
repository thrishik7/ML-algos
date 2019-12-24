import numpy as np 

class Neural_Network(object):

   def __init__(self) : #constructor
    self.inputLayerSize= 2    
    self.outputLayerSize=1
    self.hiddenLayerSize=3

    self.W1= np.random.randn(self.inputLayerSize, self.hiddenLayerSize) #(2,3)
    self.W2= np.random.randn(self.hiddenLayerSize,self.outputLayerSize) #(3,1)
   
   def sigmoid(self,z):
       return 1/(1+np.exp(-z))
   def forward(self,X):
       self.z2= np.dot(X,self.W1) #(100,2) (2,3) =(100,3)
       self.a2= self.sigmoid(self.z2)
       self.z3= np.dot(self.a2,self.W2) #(100,3) (3,1)= (100,1)
       yhat=self.sigmoid(self.z3)
       return yhat
   def costFunction(self,X,y):
       self.yhat =self.forward(X)
       J =0.5*sum(np.square((y-self.yhat)))
       return J
   def sigmoidPrime(self,z): #FOR BACKPROP
       return np.exp(-z)/(np.square(1+np.exp(-z)))
   def costFuctionPrime(self,X,y):   #FOR BACKPROP
       self.yHat =self.forward(X)
       delta3 =np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))    
       djdW2= np.dot(self.a2.T,delta3)  

       delta2= np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
       djdW1=np.dot(X.T,delta2)
       
       return djdW1, djdW2

#X= np.array(([3,5],[5,1],[10,2]), dtype=float)
#y= np.array(([75],[82],[93]), dtype=float)

X=np.random.rand(100,2)
y=np.random.rand(100,1)

nn= Neural_Network()
max_iterations= 10000
iter=0
learningRate = 0.01

while iter< max_iterations:
    djdW1, djdW2 = nn.costFuctionPrime(X,y)
    
    nn.W1= nn.W1 - learningRate*djdW1
    nn.W2= nn.W2 - learningRate*djdW2
    
    if iter %1000 ==0:
        print(nn.costFunction(X,y))
    iter= iter+1


print ("________________________________________________________")
print(y)
print ("________________________________________________________")
print(nn.forward(X))