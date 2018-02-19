import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ANN:
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def dsigmoid(x): 
        return ANN.sigmoid(x)*(1-ANN.sigmoid(x))
        
    def back_prop_learning(self, X, y, alpha,T, network = {}):
        
        n_inp_nodes = len(X.columns)
        n_out_nodes = y.nunique()
        #n_out_nodes = 4
        n_hidden = self.h
        n_units = self.s
        
        output_vector = np.zeros(y.nunique())
        #output_vector = np.zeros(4)
        
        output_mappings = {}
        
        for val in y.unique():
            output_vector = np.zeros(y.nunique())
            output_vector[val] += 1
            output_mappings[val] = output_vector
        

        #network[1]...network[n_hidden+1] 
        #network[i] is a list of lists representing weights linking nodes of ith and (i+1)th layer 
        #network[i] contains m (# neurons in next layer) lists with 
        #n elements (where n is the # neurons in current layer)
        
        xavier_int = np.sqrt(6/(n_units + n_inp_nodes))
        network[1] = np.random.uniform(-xavier_int, xavier_int, (n_units, n_inp_nodes))
        self.bias = {}
        #biases are assigned from layer 2 to last (output) layer
        self.bias[2] = np.random.uniform(-xavier_int, xavier_int, n_units)

        for layer in range(2, n_hidden+2):
            if (layer == n_hidden+1):
                xavier_int = np.sqrt(6/(n_units + n_out_nodes))
                self.bias[layer+1] = np.random.uniform(-xavier_int, xavier_int,n_out_nodes)
                network[layer] = np.random.uniform(-xavier_int, xavier_int, (n_out_nodes, n_units))
            else:
                xavier_int = np.sqrt(6/(n_units + n_units))
                self.bias[layer+1] = np.random.uniform(-xavier_int, xavier_int, n_units)
                network[layer] = np.random.uniform(-xavier_int, xavier_int, (n_units, n_units))
        
        #Finished initializing the weights

        #The following is the initialization for testing with David's example
        
#        classes = [0,1,2,3]
#        for val in classes:
#            output_vector = np.zeros(4)
#           output_vector[val] += 1
#            output_mappings[val] = output_vector
#        self.bias = {}
#        self.bias = {2: np.array([-1.14282988,  0.22379589,  1.07941935, -1.55562539,  0.05029518]), 
#                         3: np.array([ 0.28503219, -0.28988183, -2.15749797,  0.21010699])}
        
#        network = {}
#         network = {1: np.array([[-0.45253739, -0.16098125, -0.08067616],
#        [-0.20108552, -0.23830988, -0.10435417],
#        [ 0.23856118, -0.41676105, -0.84241045],
#        [-0.48465322,  0.68205681, -0.08747013],
#        [-0.60252568,  0.61336393,  0.15432761]]), 
#                        2: np.array([[ 0.51830356, -0.38052089, -0.11475257, -0.638776  , -0.46862776],
#        [-0.5736774 , -0.07588861,  0.12875949,  0.64619578, -0.63165241],
#        [ 0.2970735 , -0.33831311, -0.52670153,  0.02932394,  0.64146576],
#        [ 0.55335698,  0.40126154, -0.04065573, -0.09815836,  0.52545504]])}
        J = []
        for iteration in range(T):
            loss = 0
            for index, example in X.iterrows():
                #forward prop
                a = {}
                z = {}
                a[1] = example.values
                #z[1] = a

                for layer in range(2, n_hidden+3):
                    z[layer] = network[layer-1].dot(a[layer-1]) + self.bias[layer]
                    a[layer] = ANN.sigmoid(z[layer])
                    #print('Pre-activated layer',layer, ":", z[layer])
                    #print('Activated layer',layer, ":", a[layer])
                    
                    
                err = {}
                #Back-prop output layer
                k = n_hidden + 2 #Output layer
                err[k] = output_mappings[y[index]] - a[k]
                #print(err[k])
                deltas = {}
                deltas[k] = np.array(ANN.dsigmoid(z[k]) * err[k])



                for j in range(k-1, 1, -1):
                    #print(network[j])
                    deltas[j] = ANN.dsigmoid(z[j]) * network[j].transpose().dot(deltas[j+1])


                for layer in range(1,n_hidden+2):
                    self.bias[layer + 1] = self.bias [layer + 1] + alpha * deltas[layer + 1] 
                    network[layer] = network[layer] + alpha*np.outer(deltas[layer+1], a[layer])
                    #print("Updated layer",layer,"weights" , network[layer])

                loss = loss + np.mean((a[n_hidden+2] - output_mappings[y[index]])**2)
            J.append(loss)
            
        plt.title("Training curve")
        plt.xlabel("No. of epochs")
        plt.ylabel("Total error on training set")
        plt.plot(J)
        plt.show()
        self.network = network
#        print(network)
        return network
        
        
    def __init__(self, h, s):
        if (h<=0):
            raise ValueError('Need at least one hidden layer')
            
        self.h = h
        self.s = s
        
    def fit(self, X, y, alpha, T):
        model = self.back_prop_learning(X, y, alpha, T)
        
        #return model 
    
    def predict(self,T):
        network = self.network
        n_hidden = self.h
        pred_probs = []
        for index, example in T.iterrows():
            a = {}
            z= {}
            a[1] = example.values
                #z[1] = a
            for layer in range(2, n_hidden+3):
                    z[layer] = network[layer-1].dot(a[layer-1]) + self.bias[layer]
                    a[layer] = ANN.sigmoid(z[layer])
#             print(z[2])
#             print(a[3])
            pred_probs.append(a[layer])
        return pred_probs
            
    def predict_values(self,T):
        network = self.network
        n_hidden = self.h
        preds = []
        for index, example in T.iterrows():
            a = {}
            z= {}
            a[1] = example.values
                #z[1] = a
            for layer in range(2, n_hidden+3):
                    z[layer] = network[layer-1].dot(a[layer-1]) + self.bias[layer]
                    a[layer] = ANN.sigmoid(z[layer])
#             print(z[2])
#             print(a[3])
            preds.append(np.argmax(a[layer]))
        return preds
        
        
    def print_nn(self):
        nodes = 0
        weights = self.network
        for layer in weights.keys():
            n = len(weights[layer]) #Number of nodes in next layer
            m = len(weights[layer][0]) #Number of nodes in current layer
            
            for j in range(nodes + 1+m, nodes + m+1+n):
                    print('w[',0, j, '] = ', self.bias[layer+1][j-nodes-1-m])
            for i in range(nodes + 1, nodes + m+1):
                for j in range(nodes + 1+m, nodes + m+1+n):
                    print('w[',i, j, '] = ', self.network[layer][j-nodes-1-m][i-nodes-1])
                
            nodes += m 
        #print(self.network)
            
        
if __name__ == "__main__":
	df = pd.read_csv('train.csv')
	print("Loaded dataset")
	train, test = train_test_split(df, test_size = .7)
	X_train = train.drop('label', 1)
	y_train = train.label

	X_test = test.drop('label', 1)
	y_test = test.label

	#Initializing neural network
	start = time.time()
	ann = ANN(1, 70)
	ann.fit(X_train,y_train,0.001, 70)
	print("Time taken :", time.time()- start, 'seconds')

	#Training accuracy
	train_preds = ann.predict_values(X_train)
	train_accuracy = accuracy_score(y_train, train_preds)
	print("Training accuracy:", train_accuracy)

	#Testing accuracy
	test_preds = ann.predict_values(X_test)
	test_accuracy = accuracy_score(y_test, test_preds)
	print("Testing accuracy:", test_accuracy)