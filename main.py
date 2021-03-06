from pyspark import SparkContext
import loadFiles_v1 as lf
import numpy as np
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from functools import partial
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB


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


def trainedModels(groupedPoint):
	model = GaussianNB()
	x = [points.features for points in groupedPoint[1]]
	y = [points.label for points in groupedPoint[1]]
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
	svector = SparseVector(len(dictionary),vector_dict)
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


############################### Split Dataset ###############################
data_O,Y_O=lf.loadLabeled("./data/train")
data,x_test,Y,y_test = cross_validation.train_test_split(data_O, Y_O, test_size=0.2, random_state=0)
############################### Split Dataset ###############################


############################# Creat Dictionary #############################
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
model=NaiveBayes.train(labeledRDD)
mb=sc.broadcast(model)
predictions=sc.parallelize(x_test).map(partial(Predict,dictionary=dict_broad.value,model=mb.value)).collect()
print "Naive_Bayes_Accuracy:",Accuracy([i[1] for i in predictions],y_test)
############################### Train NBmodel ###############################


############################### Train Gaussian NBmodel ###############################
data_class=zip(data,Y)
dcRDD=sc.parallelize(data_class,numSlices=16)
labeledRDD=dcRDD.map(partial(createBinaryLabeledPoint,dictionary=dict_broad.value))
labeledRDD=labeledRDD.map(lambda x: (randint(1,5),x))
groupRDD=labeledRDD.groupByKey().mapValues(list)
models = groupRDD.map(trainedModels).collect()
mbs=sc.broadcast(models)
predictions = sc.parallelize(x_test).map(partial(groupPredict,dictionary=dict_broad.value,models=mbs.value)).collect()
print "GaussianNB_Accuracy:",Accuracy(predictions,y_test)
############################### Train Gaussian NBmodel ###############################


# Accuracy of given algo: 85.54%
# update 1.1.1

