import findspark
findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import udf
from pyspark.sql.functions import split 
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.regression import LinearRegression
import csv 

#creation of Spark shell
try:
    sc = ps.SparkContext('local[4]')
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")
"""
#TF-DF:
#The steps are :
1. Tokenise input
2. Remove stop words
3. Find TF through Hashing 
4. Find IDF
5. Create pipeline and return result
"""
def createTFIDFFeatures(inputData,numOfFeatures=300,inputColumn="review", outputColumn="result") :
  tokenizer = Tokenizer(inputCol=inputColumn, outputCol="words")
  remover = StopWordsRemover(inputCol="words", outputCol="filtered")
  hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=numOfFeatures)
  idf = IDF(inputCol="rawFeatures", outputCol=outputColumn)
  pipeline = Pipeline(stages=[tokenizer,remover, hashingTF,idf])
  model = pipeline.fit(inputData)
  return model.transform(inputData).drop("words").drop("rawFeatures")

"""
We find the sentiment of each article here. We first download the article through the 
Nespaper3k API, and then parse it. The NLTK  sentiment analyser is used to calculate 
a compound sentiment.

"""
def func(url):
    try:
      article = Article(url)
      article.download()
      article.parse()
      sia = SIA()
      pol_score = sia.polarity_scores(article.text)
      return pol_score['compound']
    except Exception as e:
      print(e)
"""
Change data.csv filepath when running. 

The CSV file is read and the TF-IDF features are calculated for the Themes
The polarity of all the article is calculated 
The tones given in the GDELT database is split into 4 features
The final feature vector is collected through vector assembler into the output column features
The column '_6' contains the target values of the Dow Jones
The dataset is split into 70:30 train:test ratio.
The linear regression model is trained and the RMSE, R2 scores are computed.
"""
df = sqlContext.read.csv("data.csv")
df1=createTFIDFFeatures(df, inputColumn="_c2")
sample_udf = udf(lambda x: func(x), FloatType())
df1=df1.withColumn("polarity",sample_udf(df["_c7"]))
split_col = split(df1['_c6'], ',')
df1 = df1.withColumn('tone1', split_col.getItem(0).cast('float'))
df1 = df1.withColumn('tone2', split_col.getItem(1).cast('float'))
df1 = df1.withColumn('tone3', split_col.getItem(2).cast('float'))
df1 = df1.withColumn('tone4', split_col.getItem(3).cast('float'))
def extract(row):
    if(row._c10=="#N/A"):
        return (row.polarity,)+(row.tone1,)+(row.tone2,)+(row.tone3,)+(row.tone4,)+(float('24000'),)+tuple(row.result.toArray().tolist())
    else:
        return (row.polarity,)+(row.tone1,)+(row.tone2,)+(row.tone3,)+(row.tone4,)+(float(row._c10),)+tuple(row.result.toArray().tolist())
df1=df1.rdd.map(extract).toDF(["result"])
from pyspark.ml.feature import VectorAssembler
cols=df1.columns
cols.remove('_6')
cols.remove('result')
vectorAssembler = VectorAssembler(inputCols = cols, outputCol = 'features')
fea = vectorAssembler.transform(df1)
fea = fea.select(['features', '_6'])
splits = fea.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='_6', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)