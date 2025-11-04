import numpy as np
import random
import pickle

#Activation functions candidates
@staticmethod
def logistic_sigmoid(s):
    return 1/(1+np.exp(-s))
 
@staticmethod
def tanh(s):
    return 2/(1+np.exp(-s))-1
 
def soft_plus(s):
    return np.log(1+np.exp(s))
  
def ReLU(s):
    return np.maximum(0,s)
   
#Derivatives of Activation functions
def deriv_logistic_sigmoid(s):
    return logistic_sigmoid(s)*(1-logistic_sigmoid(s))

def deriv_tanh(s):
    return (1-tanh())/2

def deriv_soft_plus(s):
    return 1/(1+np.exp(-s))

def deriv_ReLU(s):
    return (s>0)*1

    


#MSE 사용
def cost(y_pred, y):
    return np.mean((y_pred-y)**2)


#derivative by y_pred
def cost_derivative(y_pred, y):
    return (y_pred-y)*(2/y_pred.shape[0])



class MLP:
    #input # : 4
    #output # : 20
    def __init__(self, batch_size, input_dim, hidden1_dim, hidden2_dim, output_dim, activation_fn, activation_deriv, init_method = 'He', learning_rate =0.01):
        self.batch_size = batch_size
        self.lr = learning_rate

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        self.activation_name = activation_fn.__name__
        self.init_method = init_method

        self.activation = activation_fn
        self.activation_derivative = activation_deriv

        
        if init_method == 'He':
            scale1 = np.sqrt(2./input_dim)
            scale2 = np.sqrt(2./hidden1_dim) 
            scale3 = np.sqrt(2./hidden2_dim)
        
        elif init_method == 'Xavier':
            scale1 = np.sqrt(1./input_dim)
            scale2 = np.sqrt(1./hidden1_dim) 
            scale3 = np.sqrt(1./hidden2_dim)

        else:
            scale1, scale2, scale3 = 0.01,0.01,0.01


        self.w1 = np.random.randn(input_dim, hidden1_dim)*scale1
        self.b1 = np.zeros(hidden1_dim)

        self.w2 = np.random.randn(hidden1_dim, hidden2_dim)*scale2
        self.b2 = np.zeros(hidden2_dim)

        self.w3 = np.random.randn(hidden2_dim, output_dim)*scale3
        self.b3 = np.zeros(output_dim)

        self.cache = {}
        self.gradient = {}

    
    def forward(self, X):
        #Hidden1
        zsum1 = X @ self.w1 + self.b1
        z1 = self.activation(zsum1)

        #Hidden2
        zsum2 = z1 @ self.w2 + self.b2
        z2 = self.activation(zsum2)

        #Output => Linear 방식 사용
        Osum = z2 @ self.w3 + self.b3
        y_pred = Osum

        self.cache = {'X': X, 'z1' : z1,'zsum1' : zsum1, 'z2' : z2, 'zsum2' : zsum2}

        return y_pred

    def backward(self,y_pred,y):
        dJ_dy_pred = cost_derivative(y_pred,y)
        dJ_dO = dJ_dy_pred #출력층은 linear하게 설정해서, chainrule을 썼을 때, 앞에만 남음.
        dw3 = self.cache['z2'].T @ dJ_dO
        db3 = np.sum(dJ_dO, axis = 0)    #개별 오차를 합쳐주기 위함
#--------------------------------------------------------------------------------------

        dJ_dz2 = dJ_dO @ self.w3.T
        dJ_dzsum2 = dJ_dz2*self.activation_derivative(self.cache['zsum2'])
        dw2 = self.cache['z1'].T @ dJ_dzsum2
        db2 = np.sum(dJ_dzsum2, axis=0)

#--------------------------------------------------------------------------------------

        dJ_dz1 = dJ_dzsum2 @ self.w2.T
        dJ_dzsum1 = dJ_dz1*self.activation_derivative(self.cache['zsum1'])
        dw1 = self.cache['X'].T@ dJ_dzsum1
        db1 = np.sum(dJ_dzsum1, axis = 0)

        self.gradient = {'dw1' : dw1, 'db1' : db1, 'dw2' : dw2, 'db2' : db2, 'dw3' : dw3, 'db3' : db3}



    def update_parameters(self):
        self.w1 = self.w1 - self.lr * self.gradient['dw1']
        self.b1 = self.b1 - self.lr * self.gradient['db1']
        self.w2 = self.w2 - self.lr * self.gradient['dw2']
        self.b2 = self.b2 - self.lr * self.gradient['db2']
        self.w3 = self.w3 - self.lr * self.gradient['dw3']
        self.b3 = self.b3 - self.lr * self.gradient['db3']

    
    def train(self, X, y, epochs):
        print("-------Train Start!-----------")
        print(f"-------BatchSize : {self.batch_size}-------")
        print(f"-------Activation : {self.activation_name}-------")
        print(f"-------Initialization : {self.init_method}-------")

        num_samples = X.shape[0]

        
        for i in range(epochs):
            # shuffling
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            epoch_loss_sum = 0

            for j in range(0, num_samples, self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]

                y_pred = self.forward(X_batch)

                loss = cost(y_pred,y_batch)
                epoch_loss_sum += loss*X_batch.shape[0]

                self.backward(y_pred,y_batch)
                self.update_parameters()

            if i % 100 ==0:
                avg_epoch_loss = epoch_loss_sum / num_samples
                print(f"{i}th epoch, Average Loss: {avg_epoch_loss:.6f}")

        print("Train Finished.")


    def predict(self, X):
        return self.forward(X)


    #Saving & Loading Weights
    def save_weights(self, filename="model2_weights.pkl"):
        weights = {
            'w1' : self.w1, 'b1' : self.b1,
            'w1' : self.w2, 'b1' : self.b2,
            'w1' : self.w3, 'b1' : self.b3
        }
        with open(filename, 'wb') as f:
            pickle.dump(weights,f)
        print(f"Model2's weights are saved on {filename}.")

    def load_weights(self, filename = "model2_weights.pkl"):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        self.w1, self.b1 = weights['w1'], weights['b1']
        self.w2, self.b2 = weights['w2'], weights['b2']
        self.w3, self.b3 = weights['w3'], weights['b3']

        print("Model2's weights are loaded")

    
