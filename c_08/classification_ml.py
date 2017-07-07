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

    from pyspark.sql import Row
    from pyspark.sql import SQLContext

    from pyspark.ml.linalg import Vectors
    from pyspark.ml.classification import LogisticRegression

    from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator


    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

"""
https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html
https://spark.apache.org/docs/2.1.0/ml-tuning.html
"""


sc = SparkContext('local')
sqlContext = SQLContext(sc)

##############################################DATA######################################################################


def save_model(model, model_name):
    output_dir = model_name
    shutil.rmtree(output_dir, ignore_errors=True)
    model.save(sc, output_dir)


print('*' * 50, 'DATA', '*' * 50)
iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_y = [(int(Y[index]), Vectors.dense(X[index])) for index in range(0, X.shape[0])]
X_y = sqlContext.createDataFrame(X_y, ["label", "features"])
train, test = X_y.randomSplit([0.9, 0.1])
print(train.show())

##############################################MODELS_TRAIN##############################################################
print('*' * 50, 'MODELS_TRAIN', '*' * 50)
lr = LogisticRegression(maxIter=10, regParam=0.01)

paramMap = ({lr.regParam: 0.1, lr.threshold: 0.55, lr.maxIter: 100, })

paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.threshold, [0.51, 0.56])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

model = tvs.fit(X_y) # lr.fit(train, paramMap)

################################################TESTING_MODEL###############################################################
print('*' * 50, 'TESTING_MODEL', '*' * 50)
predictions = model.transform(test)
result = predictions.select("features", "label", "prediction").collect()
for row in result:
    print("features=%s, label=%s -> prediction=%s"
          % (row.features, row.label, row.prediction))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Acc = %g " % (accuracy))
