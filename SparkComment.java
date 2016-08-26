import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;


public class SparkComment {
	static LexicalizedParser lp = LexicalizedParser.loadModel("edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz");		//加载z中文分词库
	public  String focus = new String();  	//评价对象
	public  String relation = new String();	//匹配关系
	public List<Map.Entry<String,Integer>> result = new ArrayList<Map.Entry<String,Integer>>();
	public String filename = new String();
	public SparkComment(String focus, String filename){
		this.focus = focus;
		this.relation = "nsubj";
		this.result = this.SparkAnalyse();
	}
	public SparkComment(){
		this.focus = "价格";
		this.relation = "nsubj";
		this.result = this.SparkAnalyse();
	}
	public  List<Map.Entry<String,Integer>> SparkAnalyse(){
		long startTime = System.currentTimeMillis();
		SparkConf conf = new SparkConf();
		conf.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext("local[*]", "Spark", conf);
		JavaRDD<String> lines = sc.textFile(this.filename);		//		读取测试文件
		JavaRDD<String> focusRDD = lines.filter(new focusPoint(this.focus)); 	//　过滤含有关注点的句子
		JavaRDD<String> segRDD = focusRDD.map(new Seg());			//		将句子分词
		JavaRDD<String> parseRDD = segRDD.map(new Parse(this.focus, this.relation)); 	// 	构建依存关系树
		JavaPairRDD<String, Integer> ones = parseRDD.mapToPair(new MapOne()); 	//		映射每一个元素 
		JavaPairRDD<String, Integer> counts = ones.reduceByKey(new Count());	//		统计每一个元素	
		List<Tuple2<String,Integer>> result = counts.collect();
		Map<String, Integer> map = new HashMap<String, Integer>();	
		for(Tuple2<String, Integer> tuple: result){
			map.put(tuple._1(), tuple._2());
		}
		List<Map.Entry<String,Integer>> infolds = new ArrayList<Map.Entry<String, Integer>>(map.entrySet());  // 转化为list，然后对值排序
		Collections.sort(infolds, new Comparator<Map.Entry<String, Integer>>(){
			public int compare(Map.Entry<String,Integer>o1, Map.Entry<String, Integer>o2){
				return (o2.getValue() - o1.getValue());
				}
			});
		long endTime = System.currentTimeMillis();
		long Time = endTime - startTime;
		System.out.println(Time);
		sc.stop();
		return infolds;
	}

	 static class focusPoint implements Function<String, Boolean>{
		 String focus = new String();
		 public focusPoint(String focus){
			 this.focus = focus;
		 }
		public Boolean call(String x) throws Exception{
			return x.contains(focus);		//	　过滤出含有关注点的词语
		}
	}
	static class Seg implements Function<String, String>{
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		public String call(String sentence) throws Exception {
			String segStr = "";
			Segment segment = HanLP.newSegment();	
			List<Term> termList = segment.seg(sentence);		// 分词
			StringBuilder sb = new StringBuilder();
			for(Term term: termList){
				sb.append(term.word + " ");
			}
			segStr = sb.toString();
			// TODO Auto-generated method stub
			return segStr;		// 	样例:" 这个 东西 挺 不错"
		}
	}
	static class MapOne implements PairFunction<String, String, Integer>{
		/*
		 *  将每一个元素映射出每一个pair
		 * @see org.apache.spark.api.java.function.PairFunction#call(java.lang.Object)
		 */
		public Tuple2<String, Integer> call(String s){
			return new Tuple2<String, Integer>(s, 1);
		}
	}
	
	static class Parse implements Function<String, String>{
		String focus = new String();
		String relation = new String();
		public Parse(String focus, String relation){
			this.focus = focus;
			this.relation = relation;
		}
		public String call(String  sentence) throws Exception {
			String[] words = sentence.split(" ");
			List<CoreLabel> rawWords = new ArrayList<CoreLabel>();		//		构建词汇库
			for(String word : words){
				CoreLabel l = new CoreLabel();
				l.setWord(word);
				rawWords.add(l);
			}
			
			//形成依存关系树
			Tree parse = lp.apply(rawWords);
			TreebankLanguagePack tlp = lp.getOp().langpack();
			GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
			GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
			List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
			
			for(TypedDependency td: tdl){
				String pair = td.toString();
				if(pair.contains(focus) && pair.contains(relation)){
					pair = pair.replaceAll("\\d+","");		//去除数字
					pair = pair.replaceAll("[\\pP]","");		//去除标点符号
					pair = pair.replaceAll(focus, "");		//去除关注点
					pair = pair.replaceAll(relation, "");  	//删除匹配关系
					return pair;
				}
			}
			return "";
 		}
	}
	static class Count implements Function2<Integer, Integer, Integer>{
		/*
		 * 计算每一个元素
		 */
		public Integer call(Integer i1, Integer i2){
			return i1 + i2;
		}
	}
	public List<Map.Entry<String,Integer>> getResult(){
		return this.result;
	}
	public static void main(String[] agrs){
		String focus = "价格";
		String filename = "/home/quincy1994/forwork/SparkComment/pinglun.txt";
		SparkComment sc = new SparkComment(focus, filename);
		List<Map.Entry<String,Integer>> result = sc.getResult();
		System.out.println("------与价格相关的评价语统计排序结果--------");
		for(int i=0; i< result.size();i++){
			String id = result.get(i).toString();
			System.out.println(id);
		}
	}
} 
