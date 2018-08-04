from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import random
import math
import sklearn.metrics as metrics

class ValentiMLP(BaseEstimator, ClassifierMixin):
    
    # Initialise model hyperparameters.
    # eta : learning rate
    # alpha : momentum
    # lambda_ : weight decay
    # n_epoch : number of max training epoch
    # n_batch : number of batch to split the training set
    # n_input, n_hidden, n_output : number of input, hidden, and output units.
    # task : either "regression" or "classification". 
    #        When the value is "classification", output units will be sigmoidal (tanh).
    #        When the value is "regression", output units will be linear. 
    def __init__(self, task, n_input=0, n_hidden=0, n_output=0, eta=0, alpha=0, lambda_=0, n_epoch=100, n_batch=1):
        if task != "classification" and task != "regression":
            raise Exception("Unknown task type.")
        
        self.tol = 1e-3 # tolerance for early stopping during training
        self.task = task
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.eta = eta
        self.alpha = alpha
        self.lambda_ = lambda_
        self.n_epoch = n_epoch
        self.n_batch = n_batch

    # Create and initialise connection weights.
    def create_network(self, n_input, n_hidden, n_output):
        hidden_range = 1 / math.sqrt(n_input) #math.sqrt(6) / sqrt(n_input + n_hidden)
        output_range = 1 / math.sqrt(n_hidden) #math.sqrt(6) / sqrt(n_hidden + n_output)
        
        hidden_layer = [
            {
                "weights" : [ random.uniform(-hidden_range, hidden_range) for _ in range(n_input + 1)],
                "updates" : [ 0 for _ in range(n_input + 1)],
                "momentum": [ 0 for _ in range(n_input + 1)]
            }
            for _ in range(n_hidden)
        ]
        output_layer = [
            {
                "weights" : [ random.uniform(-output_range, output_range) for _ in range(n_hidden + 1)],
                "updates" : [ 0 for _ in range(n_hidden + 1)],
                "momentum": [ 0 for _ in range(n_hidden + 1)]
            }
            for _ in range(n_output)
        ]

        return [hidden_layer, output_layer]

    # Computes the mean squared error between two vectors.
    def mse(self, v1, v2):
        error = 0
        
        if len(v1) != len(v2):
            raise Exception("MSE: Data dimensions do not agree.")

        for i in range(len(v1)):
            error += (v1[i] - v2[i])**2
        
        return error
    
    # Computes the mean euclidean error between two vectors. 
    def mee(self, v1, v2):
        error = 0
        
        if len(v1) != len(v2):
            raise Exception("MEE: Data dimensions do not agree.")
        
        for i in range(len(v2)):
            error += (v1[i] - v2[i])**2
        
        return math.sqrt(error)

    # Computes a neuron's activation.
    def activation(self, weights, inputs):
        activation = weights[-1] # bias

        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        
        return activation

    # Sigmoid transfer function.
    def transfer_sigmoid(self, activation):
        return 1 / (1 + math.exp(-activation))
    
    # Hyperbolic tangent transfer function.
    def transfer_tanh(self, activation):
        return math.tanh(activation)

    # Sigmoid's grandient.
    def gradient_sigmoid(self, output):
        return output * (1 - output)

    # Hyperbolic tangent's grandient.
    def gradient_tanh(self, output):
        return 1 - output ** 2

    def transfer(self, activation):
        return self.transfer_tanh(activation)

    def gradient(self, output):
        return self.gradient_tanh(output)

    # Computes a forward pass, starting from an input sample.
    def forward_pass(self, inputs):
        network = self._network
        task = self.task

        current = inputs
        n_layer = len(network)

        for i in range(n_layer):
            layer = network[i]
            new = []
            
            for neuron in layer:
                # output layer, regression task => linear units (linear transfer function)
                if task == "regression" and i == n_layer - 1: 
                    neuron["output"] = self.activation(neuron["weights"], current)
                else: #hidden layer
                    neuron["output"] = self.transfer(self.activation(neuron["weights"], current)) 
                new.append(neuron["output"])
            
            current = new
        return current

    # Computes error backpropagation.
    # Assumes the forward pass has already been computed.
    def backpropagation_pass(self, target):
        network = self._network
        task = self.task
        n_layer = len(network)

        for i in reversed(range(n_layer)):
            layer = network[i]

            for j in range(len(layer)):
                neuron = layer[j]
                error = 0

                if i == n_layer - 1: # output layer
                    error = target[j] - neuron["output"]
                    
                    if task == "regression":
                        neuron["delta"] = error
                    else:
                        neuron["delta"] = error * self.gradient(neuron["output"])    

                else: # hidden layer 
                    for fwd_neuron in network[i + 1]:
                        error += fwd_neuron["weights"][j] * fwd_neuron["delta"]
                    neuron["delta"] = error * self.gradient(neuron["output"])

    # Computes weights updates.
    # Assumes the backpropagation pass has already been computed.
    def compute_updates(self, tr_sample):
        network = self._network
        eta = self.eta
        inputs = []
        
        for i in range(len(network)):
            if i == 0:
                inputs = tr_sample
            else:
                inputs = [ neuron["output"] for neuron in network[i - 1] ]
            
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron["updates"][j] += eta * neuron["delta"] * inputs[j]
                
                neuron["updates"][-1] += eta * neuron["delta"] # bias

    # Perform weight updates, applying momentum and weight decay (regularization).
    def update_weights(self, batch_len):
        network = self._network
        lambda_ = self.lambda_
        alpha = self.alpha
        
        for layer in network:
            for neuron in layer:
                for i in range(len(neuron["weights"])):
                    neuron["updates"][i] /= batch_len
                    neuron["weights"][i] -= lambda_ * neuron["weights"][i] # regularization
                    neuron["weights"][i] += neuron["updates"][i] + neuron["momentum"][i]
                    neuron["momentum"][i] = alpha * neuron["updates"][i]
                    neuron["updates"][i] = 0


    # Split dataset into mini-batches.
    def batchify(self, dataset, targets, n_batch):
        
        if n_batch > len(dataset):
            raise Exception("Too many batches required.")

        if len(targets) != len(dataset):
            raise Exception("Data dimensions don't agree.")        
        
        shp_data = np.array(dataset).shape
        shp_tgt  = np.array(targets).shape

        rest = shp_data[0] % n_batch

        batches_data   = np.array(dataset[:-rest or None]).reshape(n_batch, -1, shp_data[1]).tolist()
        batches_target = np.array(targets[:-rest or None]).reshape(n_batch, -1, shp_tgt[1]).tolist()

        if rest != 0:
            for i in range(len(dataset[-rest:])):
                batches_data[i].append(dataset[-rest:][i])
                batches_target[i].append(targets[-rest:][i])
        
        return (batches_data, batches_target)

    # Perform training, keeping track of the error over an external validation set.
    def fit_with_validation(self, X, y, X_val, y_val):
        n_batch = self.n_batch
        n_epoch = self.n_epoch
        task = self.task
        self._network = self.create_network(self.n_input, self.n_hidden, self.n_output)
        data_batches, target_batches = self.batchify(X, y, n_batch)
        init_patience = 10
        tol = self.tol
        score_tr = 0
        score_val = 0
        acc_tr = 0
        acc_val = 0
        scores_tr = []
        scores_val = []
        accs_tr    = []
        accs_val   = []
        early_stop = False
        patience = init_patience
        eta = self.eta

        # Main training cycle.
        for epoch in range(n_epoch):
            for data_batch, target_batch in zip(data_batches, target_batches):
                for data, target in zip(data_batch, target_batch):
                    self.forward_pass(data)
                    self.backpropagation_pass(target)
                    self.compute_updates(data)

                self.update_weights(len(data_batch))
                
            # Compute metrics.
            score_val = self.score(X_val, y_val)
            score_tr  = self.score(X, y)

            if task == "classification":
                acc_val = self.accuracy(X_val, y_val)
                acc_tr  = self.accuracy(X, y)

            # Early stop: stop if the score's improvement is not significant (and run out of patience)
            improv = math.inf
            if epoch > 0:
                improv = (scores_tr[-1][1] - score_tr) / scores_tr[-1][1]
                if improv < tol:
                    patience -= 1
                    if patience == 0:
                        early_stop = True
                else:
                    patience = init_patience


            # Saving metrics.            
            scores_val.append((epoch, score_val))
            scores_tr.append((epoch, score_tr))
            if task == "classification":
                accs_val.append((epoch, acc_val))
                accs_tr.append((epoch, acc_tr))
            
            print("- Epoch: %4d Params: %s\n Improv: %f%% \t TR error: %.5f VAL error: %.5f Accuracy TR: %.5f Accuracy VAL: %.5f" % (epoch, str(self.get_params()), improv, score_tr, score_val, acc_tr, acc_val))
            
            if early_stop:
                break
        

        self.scores_val_ = scores_val
        self.accs_val_   =  accs_val
        self.scores_tr_ = scores_tr
        self.accs_tr_   =  accs_tr
        return self
    
    # Perform training.
    def fit(self, X, y):
        init_patience = 10
        task = self.task
        n_batch = self.n_batch
        n_epoch = self.n_epoch
        self._network = self.create_network(self.n_input, self.n_hidden, self.n_output)
        data_batches, target_batches = self.batchify(X, y, n_batch)

        tol = self.tol
        score_tr = 0
        acc_tr = 0
        scores_tr = []
        accs_tr    = []
        early_stop = False
        patience = init_patience
        eta = self.eta

        # Main training cycle.
        for epoch in range(n_epoch):            
            for data_batch, target_batch in zip(data_batches, target_batches):
                for data, target in zip(data_batch, target_batch):
                    self.forward_pass(data)
                    self.backpropagation_pass(target)
                    self.compute_updates(data)

                self.update_weights(len(data_batch))

            # Computing metrics. 
            score_tr  = self.score(X, y)
            if task == "classification":
                acc_tr  = self.accuracy(X, y)

            # Early stop.
            improv = math.inf
            if epoch > 0:
                improv = (scores_tr[-1][1] - score_tr) / scores_tr[-1][1]
                if improv < tol:
                    patience -= 1
                    if patience == 0:
                        early_stop = True
                else:
                    patience = init_patience

           # Saving metrics.
            scores_tr.append((epoch, score_tr))
            if task == "classification":
                accs_tr.append((epoch, acc_tr))
            
            #print("- Epoch: %4d Params: %s\n Improv: %f%% TR error: %.5f Accuracy TR: %.5f" % (epoch, str(self.get_params()), improv, score_tr, acc_tr))
            
            if early_stop:
                break
        
        self.scores_tr_ = scores_tr
        self.accs_tr_   =  accs_tr
        return self
    
    # Compute network's output.
    def predict(self, X):
        return [ self.forward_pass(sample) for sample in X ]
    
    # Compute networks's score.
    def score(self, X, y):
        error = 0

        for i in range(len(X)):
            prediction = self.forward_pass(X[i])
            #error += self.mse(prediction, y[i])
            error += self.mee(prediction, y[i])

        return -(error / len(X))
    
    # Compute accuracy (classification only).
    def accuracy(self, X, y):
        if self.task == "regression":
            raise Exception("Cannot predict accuracy score for regression tasks.")
        
        y_true = [ np.argmax(x) for x in y ]
        y_pred = [ np.argmax(x) for x in self.predict(X) ]
        return metrics.accuracy_score(y_true, y_pred)