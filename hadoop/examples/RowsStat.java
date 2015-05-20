import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
//import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat;

import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;

import java.util.Random;

public class RowsStat {

  public static class RowsMapper
       extends Mapper<Text, Text, Text, NullWritable>{


    protected void setup(Context context) throws IOException,
            InterruptedException {

       Configuration conf = context.getConfiguration();
    }

    public String rowFilter(String row) {
        String result = "";
        String type = "";
        if (row.length() > 7) {
            type = row.substring(0, 7);
            if (type.equals("SYSTEMS")) {
                result = row.substring(7, row.length());
            }
        }
        return result;
    }

    public void map(Text key, Text val, Context context
                    ) throws IOException, InterruptedException {
      //System.out.println(strb);
      String row = "";
      StringTokenizer strT = new StringTokenizer(val.toString(), "\t");
      for (;strT.hasMoreTokens(); ) {
        row = strT.nextToken("\t");
        String rowVal = rowFilter(row);
        if (rowVal.length() != 0) {
          context.write(new Text(rowVal), NullWritable.get());
        }
      }
    }
  }

  public static class StatReducer
       extends Reducer<Text, NullWritable, Text, Text> {

    
    //private MultipleOutputs<Text, Text> mou;

    protected void setup(Context context) throws IOException,
            InterruptedException {

      //mout = new MultipleOutputs<Text, Text>(context);
      Configuration conf = context.getConfiguration();
    }

    protected void cleanup(Context context) throws IOException,
            InterruptedException {
         //mout.close();
    }
    protected void reduce(Text key, Iterable<NullWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      long counter = 0;
      for (NullWritable val : values) {
        counter += 1;
      }
      context.write(key, new Text( Long.toString(counter)) );
    }
  }


  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    
    conf.set("data", "");
    Job job = Job.getInstance(conf, "RowsStat");
    job.setJarByClass(RowsStat.class);

    FileInputFormat.addInputPaths(job, args[0]);
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    
    job.setMapperClass(RowsMapper.class);
    job.setReducerClass(StatReducer.class);
    
    //Set io format
    job.setInputFormatClass(KeyValueTextInputFormat.class);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(NullWritable.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    job.setInputFormatClass(KeyValueTextInputFormat.class);
   
    //set multiple outputs 
    //LazyOutputFormat.setOutputFormatClass(job, TextOutputFormat.class);
    //job, named output name, OutputFormat class, key class, val class 
    //MultipleOutputs.addNamedOutput(job, "supplyPartitions", TextOutputFormat.class, Text.class, Text.class);
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
