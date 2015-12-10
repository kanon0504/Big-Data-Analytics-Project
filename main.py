from pyspark import SparkContext
<<<<<<< HEAD
import LoadFiles as lf
=======
import loadFiles_v1 as lf
>>>>>>> 7c5d5208b8490aa882cc90b09715592f46407311
import numpy as np
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from functools import partial
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
<<<<<<< HEAD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import LogisticRegressionWithSGD
import string

=======
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
>>>>>>> 7c5d5208b8490aa882cc90b09715592f46407311


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

<<<<<<< HEAD
# I change the function "predict" a little, because I use the "test data", created randomly with bernouille distribution from our train data, as my input, not the test data given by the professor.
=======
def createWeightedLabeledPoint(doc_class,dictionary):
	words=doc_class[0].strip().split(' ')
	vector_dict={}
	for w in words:
		vector_dict[dictionary[w]]=1
	return (doc_class[1], np.array(SparseVector(len(dictionary),vector_dict)))

>>>>>>> 7c5d5208b8490aa882cc90b09715592f46407311
def Predict(name_text,dictionary,model):
	words=name_text[0].strip().split(' ')
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
<<<<<<< HEAD
	return (name_text[1], model.predict(SparseVector(len(dictionary),vector_dict)))

#data,Y=lf.loadLabeled("/Users/Kanon/Documents/X-courses/MAP670/Big Data Analytics Project 2015/data/train")
data,Y=lf.loadLabeled("data/train")


#data preprocessing
data=[w.decode('utf-8') for w in data ]
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
        
for doc_id, text in enumerate(data):
    
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
    data[doc_id] = doc  




#divide train data into X_test and X_train, divide the label into y_train and y_test
flag = np.random.binomial(1, 0.75, len(data))
X_train=[data[i] for i in range(len(data)) if flag[i]]
y_train=[Y[i] for i in range(len(data)) if flag[i]]
X_test=[data[i] for i in range(len(data)) if not flag[i]]
y_test=[Y[i] for i in range(len(data)) if not flag[i]]



=======
	return (name_text[0], model.predict(SparseVector(len(dictionary),vector_dict)))

def Accuracy(predictions,y):
	counter = 0.
	for i in range(len(predictions)):
		if predictions[i] == y_test[i]:
			counter += 1.
	accuracy = counter/len(predictions)
	return accuracy

############################### Split Dataset ###############################
data_O,Y_O=lf.loadLabeled("/Users/Kanon/Documents/X-courses/MAP670/Big Data Analytics Project 2015/data/train")
data,x_test,Y,y_test = cross_validation.train_test_split(data_O, Y_O, test_size=0.2, random_state=0)
############################### Split Dataset ###############################


############################# Creat Dictionary #############################
data,Y=lf.loadLabeled("/Users/Kanon/Documents/X-courses/MAP670/Big Data Analytics Project 2015/data/train")
>>>>>>> 7c5d5208b8490aa882cc90b09715592f46407311
print len(data)
dataRDD=sc.parallelize(data,numSlices=16)
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
<<<<<<< HEAD





# NaiveBayes Model
modelNB=NaiveBayes.train(labeledRDD)
#broadcast the model
mbNB=sc.broadcast(modelNB)
name_text=zip(X_test, y_test)
#for each doc :(name,text):
#apply the model on the vector representation of the text
#return the name and the class
predictionsNB=sc.parallelize(name_text).map(partial(Predict,dictionary=dict_broad.value,model=mbNB.value)).collect()
#rate 0.8857142857142857
rateNB = sum([1.0 for i in range(len(predictionsNB)) if predictionsNB[i][0] == predictionsNB[i][1]])/len(predictionsNB)
print 'rateNB',rateNB
#0.922855773838



#SVM Model
modelSVM=SVMWithSGD.train(labeledRDD)
#broadcast the model
mbSVM=sc.broadcast(modelSVM)
name_text=zip(X_test, y_test)
#for each doc :(name,text):
#apply the model on the vector representation of the text
#return the name and the class
predictionsSVM=sc.parallelize(name_text).map(partial(Predict,dictionary=dict_broad.value,model=mbSVM.value)).collect()
#rate
rateSVM = sum(1.0 for i in range(len(predictionsSVM)) if predictionsSVM[i][0] == predictionsSVM[i][1])/len(predictionsSVM)
print 'rateSVM',rateSVM
#0.866475003993




#Logistic Regression Model
modelLR=LogisticRegressionWithSGD.train(labeledRDD)
#broadcast the model
mbLR=sc.broadcast(modelLR)
name_text=zip(X_test, y_test)
#for each doc :(name,text):
#apply the model on the vector representation of the text
#return the name and the class
predictionsLR=sc.parallelize(name_text).map(partial(Predict,dictionary=dict_broad.value,model=mbLR.value)).collect()
#rate 0.8058252427184466
rateLR = sum(1.0 for i in range(len(predictionsLR)) if predictionsLR[i][0] == predictionsLR[i][1])/len(predictionsLR)
print 'rateLR',rateLR
#0.846030985466




# We do this later, after picking up a good classifier...
'''
test,names=lf.loadUknown('/Users/Kanon/Documents/X-courses/MAP670/Big Data Analytics Project 2015/data/test')
name_text=zip(names,test)
#for each doc :(name,text):
#apply the model on the vector representation of the text
#return the name and the class
predictions=sc.parallelize(name_text).map(partial(Predict,dictionary=dict_broad.value,model=mb.value)).collect()

output=file('./classifications.txt','w')
for x in predictions:
	output.write('%s\t%d\n'%x)
output.close()
'''
=======
model=NaiveBayes.train(labeledRDD)
mb=sc.broadcast(model)
predictions=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb.value)).collect()
print "Naive_Bayes_Accuracy:",Accuracy([i[1] for i in predictions],y_test)
############################### Train NBmodel ###############################

>>>>>>> 7c5d5208b8490aa882cc90b09715592f46407311

############################### Train Gaussian NBmodel ###############################
data_class=zip(data,Y)
dcRDD=sc.parallelize(data_class,numSlices=16)
labeledRDD=dcRDD.map(partial(createWeightedLabeledPoint,dictionary=dict_broad.value)).collect()
model = GaussianNB()
predictions=model.fit(labeledRDD[1],labeledRDD[0]).predict(x_test)
print "GaussianNB_Accuracy:",Accuracy(predictions,y_test)
############################### Train Gaussian NBmodel ###############################


# Accuracy of given algo: 85.54%


