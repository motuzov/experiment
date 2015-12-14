hadoop jar wc.jar WordCount /user/analytics/wcdata /user/analytics/wcoutput
hadoop fs -ls /user/analytics/wcoutput/*
hadoop fs -cat /user/analytics/wcoutput/part-r-00000

