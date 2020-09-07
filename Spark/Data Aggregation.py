################################### PROJECT 1- EROTIMA 4 ################################################3

# importing libraries
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import DoubleType
import re
from collections import Counter
from nltk import ngrams
spark = SparkSession.builder.appName("PYSPARK").getOrCreate()

# loading dataset
df = spark.read.csv("/home/akis/Desktop/Big Data Projects/Project 1/fake_job_postings.csv",header=True)

### f.) mean and stdev of maximum salaries in fake job postings
# taking fake job postings
fake = df.na.drop().filter(f.col('fraudulent')==1) # dropping null values
# splitting salary range to 2 columns -> min_salary and max_salary
splitfake = fake.withColumn("min_salary", f.split(f.col("salary_range"), "-").getItem(0)).withColumn("max_salary", f.split(f.col("salary_range"), "-").getItem(1))
# finding mean and stddev for max_salary for fake jobs
fakestats = splitfake.select(f.mean(f.col("max_salary")).alias('mean'),f.stddev(f.col("max_salary")).alias('std')).collect()
mean = fakestats[0]['mean']
std = fakestats[0]['std']
# printing results
print(mean)
print(std)

### g.) median of minimum salary in real job postings 
# taking real job postings
real = df.na.drop().filter(f.col('fraudulent')==0) # dropping null values
# splitting salary range to 2 columns -> min_salary and max_salary
splitreal = real.withColumn("min_salary", f.split(f.col("salary_range"), "-").getItem(0)).withColumn("max_salary", f.split(f.col("salary_range"), "-").getItem(1))
# finding median for min_salary for real jobs (incorrect sort)
realstats = splitreal.orderBy("min_salary")
# we have 653 values, so median is at 327
realstats.select("min_salary").show(327) # result is 40
# finding median for min_salary for real jobs (correct sort with type casting before)
splitreal_transformed = splitreal.withColumn("min_salary",splitreal["min_salary"].cast(DoubleType()))
realstats_sorted = splitreal_transformed.orderBy("min_salary")
realstats_sorted.select("min_salary").show(327) # last value displayed is the median

### h.) detecting outliers - recomputing f.)
# fake set : removing salaries above 150k
splitfake_noOutliers = splitfake.filter(f.col('max_salary')<150000)
fakestats2 = splitfake_noOutliers.select(f.mean(f.col("max_salary")).alias('mean2'),f.stddev(f.col("max_salary")).alias('std2')).collect()
mean2 = fakestats2[0]['mean2']
std2 = fakestats2[0]['std2']
# printing results
print(mean2)
print(std2)

# recomputing g.)
# real set : removing salaries below 1k dollars and over 600k dollars
realstats_sorted_noOutliers = realstats_sorted.filter(f.col('min_salary')>999)
realstats_sorted_noOutliers = realstats_sorted_noOutliers.filter(f.col('min_salary')<600001) #611 rows left
realstats_sorted_noOutliers.select("min_salary").show(306) # last value displayed is the median

### i.) calculating most popular bigrams and trigrams
# converting to pandas for easier data manipulation
df_pandas=df.select("description","fraudulent").toPandas()
# initializing fake and real descriptions with empty lists
fake_descriptions = []
real_descriptions = []
# putting all fake and real descriptions in lists
for i in range(len(df_pandas)):
    if df_pandas["fraudulent"].iloc[i] =='1':
       fake_descriptions.append(df_pandas["description"].iloc[i]) # fake descriptions
    if df_pandas["fraudulent"].iloc[i] =='0':
       real_descriptions.append(df_pandas["description"].iloc[i]) # real descriptions

# filtering null values
fake_descriptions = list(filter(None,fake_descriptions)) 
real_descriptions = list(filter(None,real_descriptions))
# joining all descriptions in a single string
fake_descriptions = '||'.join(fake_descriptions) 
real_descriptions = '||'.join(real_descriptions) 

## finding bigrams and trigrams for fake jobs
fake_descriptions=" ".join(re.findall("[a-zA-Z]+",fake_descriptions))
f_bigrams=list(ngrams(fake_descriptions.lower().split(),2))
f_trigrams=list(ngrams(fake_descriptions.lower().split(),3))
print("Most popular bigrams in fake job postings are:")
print((Counter(f_bigrams)).most_common(10))
print("Most popular trigrams in fake job postings are:")
print((Counter(f_trigrams)).most_common(10))

## finding bigrams and trigrams for real jobs
real_descriptions=" ".join(re.findall("[a-zA-Z]+",real_descriptions))
r_bigrams=list(ngrams(real_descriptions.lower().split(),2))
r_trigrams=list(ngrams(real_descriptions.lower().split(),3))
print("Most popular bigrams in real job postings are:")
print((Counter(r_bigrams)).most_common(10))
print("Most popular trigrams in real job postings are:")
print((Counter(r_trigrams)).most_common(10))

print("End of question 1.4...")
