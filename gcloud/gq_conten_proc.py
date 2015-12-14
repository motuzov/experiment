#!/usr/bin/python
"""BigQuery I/O PySpark example."""
import json
import pyspark

sc = pyspark.SparkContext()

# Use the Google Cloud Storage bucket for temporary BigQuery export data used
# by the InputFormat. This assumes the Google Cloud Storage connector for
# Hadoop is configured.
bucket = sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')

conf = {
    # Input Parameters
    'mapred.bq.gcs.bucket': bucket,
    'mapred.bq.input.project.id': 'datascience-cni-ru',
    'mapred.bq.input.dataset.id': 'gq_content',
    #'mapred.bq.input.table.id': 'gq_content_20151102',
    'mapred.bq.input.table.id': 'gq_content_20151102_sample',
}

# Load data in from BigQuery.
table_data = sc.newAPIHadoopRDD(
    'com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat',
    'org.apache.hadoop.io.LongWritable',
    'com.google.gson.JsonObject',
    conf=conf)

def tokv(js):
    return "%s\t%s" % (js["url"], js["content"])

gq_content = (
    table_data
    .map(lambda (_, record): tokv(json.loads(record)))
    )

# Display 10 results.
print gq_content.take(10)
gq_content.saveAsTextFile("/user/gq/content")
#wco1.saveAsNewAPIHadoopDataset(conf)
