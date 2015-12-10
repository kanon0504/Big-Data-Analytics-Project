from pyspark import SparkContext
import LoadFiles as lf
import numpy as np
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from functools import partial
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from nltk.stem import PorterStemmer
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import LogisticRegressionWithSGD
import string


sc = SparkContext(appName="Simple App")


def createBinaryLabeledPoint(doc_class,dictionary):
	words=doc_class[0].strip().split(' ')
	#create a binary vector for the document with all the words that appear (0:does not appear,1:appears)
	#we can set in a dictionary only the indexes of the words that appear
	#and we can use that to build a SparseVector
	vector_dict={}
	for w in words:
		vector_dict[dictionary[w]]=1
	return LabeledPoint(doc_class[1], SparseVector(len(dictionary),vector_dict))

def createWeightedLabeledPoint(doc_class,dictionary):
	words=doc_class.strip().split(' ')
	vector_dict={}
	for w in words:
		vector_dict[dictionary[w]]=1
	return (doc_class, np.array(SparseVector(len(dictionary),vector_dict)))

def Predict(name_text,dictionary,model):
	words=name_text.strip().split(' ')
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	return (name_text, model.predict(SparseVector(len(dictionary),vector_dict)))

def Accuracy(predictions,y):
	counter = 0.
	for i in range(len(predictions)):
		if predictions[i] == y_test[i]:
			counter += 1.
	accuracy = counter/len(predictions)
	return accuracy
 
 
 
 
 
############################### Data Load & Preprocessing ###############################
data_O,Y_O=lf.loadLabeled("data/train")
#data preprocessing
data_O=[w.decode('utf-8') for w in data_O ]
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
        
for doc_id, text in enumerate(data_O):
    
    # Remove punctuation and lowercase
    punctuation = set(string.punctuation)    
    doc = ''.join([w for w in text.lower() if w not in punctuation])
     
    # Stopword removal
    doc = [w for w in doc.split() if w not in stopwords]  
    
    # Stemming
    stemmer = PorterStemmer()
    doc = [stemmer.stem(w) for w in doc] 
        
    # Covenrt list of words to one string
    doc = ' '.join(w for w in doc)
    data_O[doc_id] = doc  
############################### Data Load & Preprocessing ###############################
 
 
 

############################### Split Dataset ###############################
data,x_test,Y,y_test = cross_validation.train_test_split(data_O, Y_O, test_size=0.2, random_state=0)
############################### Split Dataset ###############################


############################# Creat Dictionary #############################
dataRDD=sc.parallelize(data_O,numSlices=16)
lists=dataRDD.map(lambda x:list(set(x.strip().split(' ')))).collect()
all=[]
for l in lists:
	all.extend(l)
dict=set(all)
print len(dict)
dictionary={}
for i,word in enumerate(dict):
	dictionary[word]=i
dict_broad=sc.broadcast(dictionary)
############################# Creat Dictionary #############################


############################### Train NBmodel ###############################
data_class=zip(data,Y)
dcRDD=sc.parallelize(data_class,numSlices=16)
labeledRDD=dcRDD.map(partial(createBinaryLabeledPoint,dictionary=dict_broad.value))
model=NaiveBayes.train(labeledRDD)
mb=sc.broadcast(model)
predictions=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb.value)).collect()
print "Naive_Bayes_Accuracy:",Accuracy([i[1] for i in predictions],y_test)
############################### Train NBmodel ###############################


############################### Train SVM model ###############################
model_SVM=SVMWithSGD.train(labeledRDD)
mb_SVM=sc.broadcast(model_SVM)
predictions_SVM=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb_SVM.value)).collect()
print "SVM_Accuracy:",Accuracy([i[1] for i in predictions_SVM],y_test)
############################### Train SVM model ###############################

############################### Train Logistic Regression model ###############################
model_LR=LogisticRegressionWithSGD.train(labeledRDD)
mb_LR=sc.broadcast(model_LR)
predictions_LR=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb_LR.value)).collect()
print "LR_Accuracy:",Accuracy([i[1] for i in predictions_LR],y_test)
############################### Train Logistic Regression model ###############################

'''
############################### Train Gaussian NBmodel ###############################
data_class=zip(data,Y)
dcRDD=sc.parallelize(data_class,numSlices=16)
labeledRDD=dcRDD.map(partial(createWeightedLabeledPoint,dictionary=dict_broad.value)).collect()
model = GaussianNB()
predictions=model.fit(labeledRDD[1],labeledRDD[0]).predict(x_test)
print "GaussianNB_Accuracy:",Accuracy(predictions,y_test)
############################### Train Gaussian NBmodel ###############################
'''

# Accuracy of given algo: 85.54%



