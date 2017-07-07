import os
import sys

import shutil

from sklearn import datasets
from sklearn.metrics import accuracy_score
import pandas as pd

# Path for spark source folder
os.environ['SPARK_HOME']="/home/mickail/spark/"

# Need to Explicitly point to python3 if you are using Python 3.x
os.environ['PYSPARK_PYTHON']="/usr/bin/python3.5"

#You might need to enter your local IP
#os.environ['SPARK_LOCAL_IP']="192.168.2.138"

#Path for pyspark and py4j
sys.path.append("/home/mickail/spark/python")
sys.path.append("/home/mickail/spark/python/lib/py4j-0.10.4-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    from pyspark.mllib.linalg import SparseVector, Matrix, Matrices
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import SVMWithSGD, SVMModel
    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
    from pyspark.mllib.util import MLUtils

    from pyspark.sql import Row
    from pyspark.sql import SQLContext

    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

sc = SparkContext('local')
sqlContext = SQLContext(sc)

#############################################BASIC_DATA_TYPES###########################################################
pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))
dm2 = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])
print('*' * 50, 'BASIC_DATA_TYPES', '*' * 50)
print(pos)
print(neg)
print(dm2)
print(sm)
##############################################MODELS_TRAIN##############################################################


def accuracy(data):
    y = [el[1] for el in data.collect()]
    y_pred = [el[0] for el in data.collect()]
    print('Accuracy:', accuracy_score(y, y_pred=y_pred))


def save_model(model, model_name):
    output_dir = model_name
    shutil.rmtree(output_dir, ignore_errors=True)
    model.save(sc, output_dir)


print('*' * 50, 'MODELS_TRAIN', '*' * 50)
iris = datasets.load_iris()
data_set = iris.data
Y = iris.target
data_set = pd.DataFrame(data_set)
data_set['labels'] = Y
print(data_set.head(5))
print(data_set.shape)

s_df = sqlContext.createDataFrame(data_set)
train_dataset = s_df.rdd.map(lambda x: LabeledPoint(x[-1], x[:4]))
training, test = train_dataset.randomSplit([0.6, 0.4])

model = NaiveBayes.train(training, 0.7)
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy(predictionAndLabel)

################################################SAVE_LOAD###############################################################
print('*' * 50, 'SAVE_LOAD', '*' * 50)
save_model(model, 'myNaiveBayesModel')
sameModel = NaiveBayesModel.load(sc, 'myNaiveBayesModel')
predictionAndLabel_1 = test.map(lambda p: (model.predict(p.features), p.label))
accuracy(predictionAndLabel_1)
