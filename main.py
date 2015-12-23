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
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import LogisticRegressionWithSGD
import string
from nltk.stem import PorterStemmer
import re



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





def createBinaryPoint(doc_class,dictionary):
	words=doc_class[0].strip().split(' ')
	#create a binary vector for the document with all the words that appear (0:does not appear,1:appears)
	#we can set in a dictionary only the indexes of the words that appear
	#and we can use that to build a SparseVector
	vector_dict={}
	for w in words:
		vector_dict[dictionary[w]]=1
	return (doc_class[1], SparseVector(len(dictionary),vector_dict))
 


 
def trainedModels(groupedPoint):
	model = GaussianNB()
	x =[points[1].toArray() for points in groupedPoint[1]]
	y = [points[0] for points in groupedPoint[1]]
	model.fit(x,y)
	return model
    
    


def Predict(name_text,dictionary,model):
	words=name_text.strip().split(' ')
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	return (name_text, model.predict(SparseVector(len(dictionary),vector_dict)))


def groupPredict(x_test,dictionary,models):
	words=x_test.strip().split(' ')
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	svector = SparseVector(len(dictionary),vector_dict).toArray()
	y_temp = []
 	for model in models:
 		y_temp.append(float(model.predict(svector)))
	return round(np.average(y_temp))


def Accuracy(predictions,y):
	counter = 0.
	for i in range(len(predictions)):
		if predictions[i] == y_test[i]:
			counter += 1.
	accuracy = counter/len(predictions)
	return accuracy



def remove_html_tags(string):
    """delete annoying html tags in the description of a book
    using a regex
    """
    return re.sub('<[^<]+?>', '', string) if string else ''


def remove_nonutf8_tags(string):
    return re.sub('[^\w]+', ' ', string)
    


    


############################### train data Loading and Preprocessing  ##############################################
data_O,Y_O=lf.loadLabeled("data/train")

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
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now','\\\\']


data_O=[remove_html_tags(w) for w in data_O ]

data_O=[remove_nonutf8_tags(w).decode('utf-8','ignore') for w in data_O ]

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
       
data_O=[w.encode('utf-8') for w in data_O ]

    
############################### train data Loading and Preprocessing  #############################################
 
 
 
############################### Split Dataset ###############################
data,x_test,Y,y_test = cross_validation.train_test_split(data_O, Y_O, test_size=0.2, random_state=0)
############################### Split Dataset ###############################

 

############################# Creat Dictionary #############################
print "Creat Dictionary_start"
print len(data)
dataRDD=sc.parallelize(data,numSlices=16).cache()
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
print "Creat Dictionary_end"
############################# Creat Dictionary #############################




############################### Models from sklearn ###############################
print " Models from sklearn_start"
############################### Train Gaussian NBmodel ###############################
data_class=zip(data,Y)
dcRDD=sc.parallelize(data_class,numSlices=16).cache()
labeledRDD=dcRDD.map(partial(createBinaryPoint,dictionary=dict_broad.value))
labeledRDD=labeledRDD.map(lambda x: (randint(1,5),x))
groupRDD=labeledRDD.groupByKey().mapValues(list)
models = groupRDD.map(trainedModels).collect()
mbs=sc.broadcast(models)
predictions = sc.parallelize(x_test).map(partial(groupPredict,dictionary=dict_broad.value,models=mbs.value)).collect()
print "GaussianNB_Accuracy:",Accuracy(predictions,y_test)
############################### Train Gaussian NBmodel ###############################
print " Models from sklearn_start_end"




############################### Models from pyspark.mllib ###############################
print "Models from pyspark.mllib_start"
data_class=zip(data,Y)
dcRDD=sc.parallelize(data_class,numSlices=16).cache()
labeledRDD=dcRDD.map(partial(createBinaryLabeledPoint,dictionary=dict_broad.value))

############ Train NBmodel 
model=NaiveBayes.train(labeledRDD)
mb=sc.broadcast(model)
predictions=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb.value)).collect()
print "Naive_Bayes_Accuracy:",Accuracy([i[1] for i in predictions],y_test)


############ Train SVM 
model_SVM=SVMWithSGD.train(labeledRDD)
mb_SVM=sc.broadcast(model_SVM)
predictions_SVM=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb_SVM.value)).collect()
print 'SVM_Accuracy:',Accuracy([i[1] for i in predictions_SVM],y_test)


############Train Logistic Regression 
model_LR=LogisticRegressionWithSGD.train(labeledRDD)
mb_LR=sc.broadcast(model_LR)
predictions_LR=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb_LR.value)).collect()
print 'RL_Accuracy:',Accuracy([i[1] for i in predictions_SVM],y_test)


print "Models from pyspark.mllib_end"
############################### Models from pyspark.mllib ###############################







###############################test data Loading and Preprocessing  ###############################
test,names=lf.loadUknown('./data/test')

test=[remove_html_tags(w) for w in test ]
test=[remove_nonutf8_tags(w).decode('utf-8','ignore') for w in test ]
for doc_id, text in enumerate(test):        
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
    test[doc_id] = doc  

test=[remove_html_tags(w).encode('utf-8') for w in test ]

############################### test data Loading and Preprocessing  #############################################



