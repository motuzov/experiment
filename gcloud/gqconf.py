gqconf = {
    # Input Parameters
    'mapred.bq.gcs.bucket': 'dataproc-522ff599-4be1-4b4f-ba9c-8258fedbd751-eu',
    'mapred.bq.input.project.id': 'datascience-cni-ru',
    'mapred.bq.input.dataset.id': 'gq_content',
    #'mapred.bq.input.table.id': 'gq_content_20151102',
    'mapred.bq.input.table.id': 'gq_content_20151102_sample',
}

# Load data in from BigQuery.

apipar = {
        'inputFormatClass':'com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat',
        'keyClass':'org.apache.hadoop.io.LongWritable',
        'valueClass':'com.google.gson.JsonObject',
        'conf': gqconf}
