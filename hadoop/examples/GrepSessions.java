import java.io.IOException;
import java.util.StringTokenizer;
//import java.text.SimpleDateFormat;
//import java.util.TimeZone;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableUtils;

import org.apache.hadoop.io.Text;
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

import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import java.util.Random;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Partitioner;
import java.lang.IllegalArgumentException;
import java.lang.Math;

import java.io.DataInput;
import java.io.DataOutput;

import java.util.regex.Pattern;
import java.util.regex.Matcher;

import java.util.ArrayList;
import java.util.List;

public class GrepSessions {


  public static class TextPair 
                      implements WritableComparable<TextPair> {
    private Text first;
    private Text second;
   
    public TextPair(String f, String s) {
      set(new Text(f), new Text(s));
    }

    public TextPair(Text f, Text s) {
      set(f, s);
    }

    public String getStr() {
      return first.toString() + "_" + second.toString();
    }

    public TextPair() {
      set(new Text(), new Text());
    }
    
    /**
     * Set the left and right values.
     */
    public void set(Text left, Text right) {
      first = left;
      second = right;
    }
    public Text getFirst() {
      return first;
    }
    public Text getSecond() {
      return second;
    }
    @Override
    public void readFields(DataInput in) throws IOException {
      first.readFields(in);
      second.readFields(in);
    }
    @Override
    public void write(DataOutput out) throws IOException {
      first.write(out);
      second.write(out);
    }
    @Override
    public int hashCode() {
      return first.hashCode() * 157 + second.hashCode();
    }
    @Override
    public boolean equals(Object right) {
      if (right instanceof TextPair) {
        TextPair r = (TextPair) right;
        return first.equals(r.first)  && second.equals(r.second);
      } else {
        return false;
      }
    }

    /** A Comparator that compares serialized TextPair. */ 
    /*ipublic static class Comparator extends WritableComparator {

      private static final Text.Comparator TEXT_COMP = new Text.Comparator();

      public Comparator() {
        super(TextPair.class);
      }

      public int compare(byte[] b1, int s1, int l1,
                         byte[] b2, int s2, int l2) {
        try {
          int firstL1 = WritableUtils.decodeVIntSize(b1[s1]) + readVInt(b1, s1);
          int firstL2 = WritableUtils.decodeVIntSize(b2[s2]) + readVInt(b2, s2);
          int cmp = TEXT_COMP.compare(b1, s1, firstL1, b2, s2, firstL2);
          if (cmp != 0) {
            return cmp;
          }
          return TEXT_COMP.compare(b1, s1 + firstL1, l1 - firstL1, b2, s2 + firstL2, l2 - firstL2); 
        }
        catch (IOException e) {
          throw new IllegalArgumentException(e);
        }
      }
    }

    static {                                        // register this comparator
      WritableComparator.define(TextPair.class, new Comparator());
    }*/

    @Override
    public int compareTo(TextPair o) {
      int cmp = first.compareTo(o.first);
      if (cmp != 0) {
        return cmp;
      } 
      return second.compareTo(o.second);
    }
  }
  
  

  public static class MyMapper
       extends Mapper<Text, Text, TextPair, Text>{

    private String strb;

    protected void setup(Mapper<Text, Text, TextPair, Text>.Context context) throws IOException,
            InterruptedException {

       Configuration conf = context.getConfiguration();
       strb = conf.get("piroutid");
       final List<Pattern> rxs = new ArrayList<>();
       StringTokenizer strT = new StringTokenizer(value.toString(), "\n");
       

    }

    public String[] SplitKey(String key) {
      String[] res = {"", ""};
      String[] fields = key.split("_");
      for (int i =0; i < fields.length; ++i) {
        if (fields.length - i <= 2 ) {
          if (fields.length - i == 1) {
            res[1] += "_";
          }
          res[1] += fields[i];
          } else {
            if (i != 0) {
              res[0] += "_";
            }
            res[0] += fields[i];
          }
        }
        return res;
    }

    public void map(Text key, Text value, Mapper<Text, Text, TextPair, Text>.Context context
                    ) throws IOException, InterruptedException {
      //System.out.println(strb);
      String[] userSession = SplitKey(key.toString());
      StringTokenizer strT = new StringTokenizer(value.toString(), "\t");
      context.write(new TextPair(userSession[0], userSession[1]), key);

    }
  }

  public static class FirstPartitioner extends Partitioner<TextPair,Text>{
    @Override
    public int getPartition(TextPair key, Text value,
                            int numPartitions) {
      return ( Math.abs(key.hashCode() * 127)) % numPartitions;
    }
  }

  public static class KeyComparator extends  WritableComparator {
    protected KeyComparator() 
    {
      super(TextPair.class, true);
    }
    @Override
    public int compare (WritableComparable w1, WritableComparable w2) {
      TextPair tp1 = (TextPair) w1;
      TextPair tp2 = (TextPair) w2;
      //int cmp = TextPair.compare(tp1.getFirst(), tp2.getFirst());
      return tp1.compareTo(tp2);
    } 
  }

  public static class GroupComparator extends WritableComparator {
    protected GroupComparator() {
      super(TextPair.class, true);
    }
    @Override
    public int compare(WritableComparable w1, WritableComparable w2) {
      TextPair tp1 = (TextPair) w1;
      TextPair tp2 = (TextPair) w2;
      return tp1.getFirst().compareTo(tp2.getFirst());
    }
  }

  public static class MergeS 
       extends Reducer<TextPair,Text,Text, Text> {

    //private IntWritable result = new IntWritable();
    
    private MultipleOutputs<Text, Text> mout;
    private String buffS;
    private long lastS = 0;

    protected void setup(Reducer<TextPair, Text, Text, Text>.Context context) throws IOException,
            InterruptedException {

      mout = new MultipleOutputs<Text, Text>(context);
    }

    protected void cleanup(Reducer<TextPair, Text, Text, Text>.Context context) throws IOException,
            InterruptedException {
         mout.close();
    }
    protected void reduce(TextPair key, Iterable<Text> values,
                       Reducer<TextPair, Text, Text, Text>.Context context
                       ) throws IOException, InterruptedException {
      int i = 0;
      String keys = new String();
      String userIdStr = first.toString();
      for (Text val : values) {
        if (i != 0) {
          keys += ";";
        }
        i += 1;
        keys += val.toString();
  
      }
      if (i > 1) {
        mout.write("supplyPartitions", new Text(usrIdStr), new Text(Integer.toString(i) + ":" + keys), "sessions");
      }
      //mout.write("supplyPartitions", key, new Text( Integer.toString(sdoubles) + "\t" + Long.toString(minLastS) + "\t" + lastSStr), "stat");
    }
  }


  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    
    Job job = Job.getInstance(conf, "GrepSessions");
    job.setJarByClass(GrepSessions.class);

    FileInputFormat.addInputPaths(job, args[0]);
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(GrepSessions.TextPair.class);
    job.setPartitionerClass(FirstPartitioner.class);
    job.setSortComparatorClass(KeyComparator.class);
    job.setGroupingComparatorClass(GroupComparator.class);

    job.setReducerClass(MergeS.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    job.setInputFormatClass(KeyValueTextInputFormat.class);
    LazyOutputFormat.setOutputFormatClass(job, TextOutputFormat.class);

    //job, named output name, OutputFormat class, key class, val class 
    MultipleOutputs.addNamedOutput(job, "supplyPartitions", TextOutputFormat.class, Text.class, Text.class);
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
