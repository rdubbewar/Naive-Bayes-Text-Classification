from __future__ import print_function
import sys
import os
from scipy.sparse import csr_matrix, hstack
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
import itertools
from sklearn.metrics import classification_report, precision_score, recall_score

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data
        self.vocab_len = data.X.shape[1]
        self.count_pos = np.zeros([1,data.X.shape[1]])
        self.count_neg = np.zeros([1,data.X.shape[1]])
        self.pos_rev = 0
        self.neg_rev = 0
        self.total_pos = 0
        self.total_neg = 0
        self.P_positive = 0.0
        self.P_negative = 0.0
        self.deno_pos = 0.0
        self.deno_neg = 0.0

        self.vocab_len = data.X.shape[1]
        self.samples = data.X.shape[0]
        self.weight = np.zeros([1,data.X.shape[1]]) # +1 for the bias
        self.for_avg_weight = 0
        self.Train(data.X,data.Y)

    def Train(self, X, Y):
        pos_indices = np.argwhere(Y == 1.0).flatten()
        neg_indices = np.argwhere(Y == -1.0).flatten()
        
        self.pos_rev = len(pos_indices)
        self.neg_rev = len(neg_indices)
        
        self.count_pos = csr_matrix.sum(X[np.ix_(pos_indices)], axis = 0 ) + self.ALPHA
        self.count_neg = csr_matrix.sum(X[np.ix_(neg_indices)], axis = 0 ) + self.ALPHA
        
        self.total_pos = csr_matrix.sum(X[np.ix_(pos_indices)])
        self.total_neg = csr_matrix.sum(X[np.ix_(neg_indices)])
        
        self.deno_pos = float(self.total_pos + self.ALPHA * X.shape[1])
        self.deno_neg = float(self.total_neg + self.ALPHA * X.shape[1])

        samples = self.samples
        valid = 0
        weight_trans = np.zeros([X.shape[1],1])
        converged = 1
        for j in range(samples):
            term = (X[j].dot(weight_trans))
            valid = 0
            if(term > 0.0) :
                valid = 1.0   
            elif term < 0.0 :
                valid =-1.0                             
            if Y[j] != valid:
                weight_trans +=  (Y[j] * X[j].transpose())
                self.for_avg_weight += Y[j]
                converged = 0  
            if converged == 1:
                break
        self.weight = weight_trans.transpose()

        return

    def PredictLabel(self, X):
        self.P_positive = log(float(self.pos_rev)) - log(float(self.pos_rev + self.neg_rev))
        self.P_negative = log(float(self.neg_rev)) - log(float(self.pos_rev + self.neg_rev))
        pred_labels = []
        
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            sum_pos = self.P_positive
            sum_neg = self.P_negative
            for j in range(len(z[0])):
                row = i
                col = z[1][j]
                occurrence = X[row , col]
                prob_pos = log((self.count_pos[0,col]))
                sum_pos = sum_pos + occurrence * prob_pos
                prob_neg = log((self.count_neg[0,col]))
                sum_neg = sum_neg + occurrence * prob_neg
            if sum_pos > sum_neg:            
                pred_labels.append(1.0)
            else:               
                pred_labels.append(-1.0)
        return pred_labels

    def LogSum(self, logx, logy):   
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    def PredictProb(self, test, indexes):
        for i in indexes:
            predicted_label = 0
            z = test.X[i].nonzero()
            sum_pos = self.P_positive
            sum_neg = self.P_negative
            for j in range(len(z[0])):
                row = i
                col = z[1][j]
                occurrence = test.X[row , col]
                prob_pos = log((self.count_pos[0,col]))
                sum_pos = sum_pos + occurrence * prob_pos
                prob_neg = log((self.count_neg[0,col]))
                sum_neg = sum_neg + occurrence * prob_neg
            if sum_pos > prob_neg:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            predicted_prob_pos = exp(sum_pos - self.LogSum(sum_pos, sum_neg))
            predicted_prob_neg = exp(sum_neg - self.LogSum(sum_pos, sum_neg))

            print(predicted_label, predicted_prob_pos, predicted_prob_neg)
            #print(predicted_label, predicted_prob_pos, predicted_prob_neg,test.X_reviews[i])

    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()

    def EvalPrecision(self, test):
        Y_pred=np.array(self.PredictLabel(test.X))
        Y_test = np.array(test.Y)
        print(precision_score(Y_test,Y_pred))
        print(recall_score(Y_test,Y_pred))

    def EvalRecall(self, test):
        Y_pred=np.array(self.PredictLabel(test.X))
        Y_test = np.array(test.Y)
        print(recall_score(Y_test,Y_pred))

class MostPosNegWord:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.vocab_len = data.X.shape[1]
        self.samples = data.X.shape[0]
        self.weight = np.zeros([1,data.X.shape[1]]) # +1 for the bias
        self.for_avg_weight = 0
        self.Train(data.X,data.Y)

    def Train(self, X, Y):
        samples = self.samples
        valid = 0
        weight_trans = np.zeros([X.shape[1],1])
        converged = 1
        for j in range(samples):
            term = (X[j].dot(weight_trans))
            valid = 0
            if(term > 0.0) :
                valid = 1.0   
            elif term < 0.0 :
                valid =-1.0                             
            if Y[j] != valid:
                weight_trans +=  (Y[j] * X[j].transpose())
                self.for_avg_weight += Y[j]
                converged = 0  
            if converged == 1:
                break
        self.weight = weight_trans.transpose()
        return
    
    def pos_neg_words(self, vocab):
        self.weight = self.weight - self.for_avg_weight
        weights = self.weight[:,1:]
        words = np.argsort(weights)
        le = words.shape[1]
        print ("Most Positive Words")
        for i in range(20):            
            print ((vocab.GetWord(words.item(0, le - 1 - i)) , weights.item(0, words.item(0, le - 1 - i))), end=" ")
        print()
            
        print ("Most Negative Words")
        for i in range(20):
            print ((vocab.GetWord(words.item(0, i)) , weights.item(0, words.item(0, i))), end=" ")
        print()

if __name__ == "__main__":
    
    print ("Reading Training Data")
    train = IMDBdata("%s/train" % sys.argv[1])
    print ("Reading Test Data")
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    nb = NaiveBayes(train, float(sys.argv[2]))

    #For accuracy
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(test))

    #For probability estimated for first 10 reviews
    alpha = float(sys.argv[2])
    print("Probability estimated for the  first 10 reviews in the test data for ALPHA = ",alpha)
    print (nb.PredictProb(test, range(10)))

    #For Precision
    nb.EvalPrecision(test)

    #For Recall
    nb.EvalRecall(test)

    #For Data preparation for most 20 positive and negative words
    row_tn = np.arange(train.X.shape[0])
    row_tt = np.arange(test.X.shape[0])
    
    col_tn = np.zeros((train.X.shape[0],))
    col_tt = np.zeros((test.X.shape[0],))
    
    bs_tn = csr_matrix((np.ones((train.X.shape[0],)),(row_tn, col_tn)), shape=(train.X.shape[0],1))
    bs_tt = csr_matrix((np.ones((test.X.shape[0],)),(row_tt, col_tt)), shape=(test.X.shape[0],1))
    
    train.X = hstack([bs_tn,train.X]).tocsr()
    test.X = hstack([bs_tt,test.X]).tocsr()
    
    mpn = MostPosNegWord(train,float(sys.argv[2]))
    mpn.pos_neg_words(train.vocab)