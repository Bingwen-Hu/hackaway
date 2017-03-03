package leaderCrawler;

// the same package
import static leaderCrawler.LeaderUtinity.*;
import static leaderCrawler.LeaderInfoCrawler.*;
import static leaderCrawler.LeaderOut.*;

// java package
import java.io.IOException;
import java.net.UnknownHostException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

// third-party package
import org.apache.http.HttpEntity;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.csvreader.CsvWriter;

/**
 * @author Mory
 * the entry of crawler
 */
public class LeaderCrawler
{
	private static ArrayList<String> linksList = new ArrayList<String>();
	private static ArrayList<String> repetitiveName = new ArrayList<String>();
	private static ArrayList<Leader> leadersList = new ArrayList<Leader>();
	private static final String baseUrlforLink = "http://baike.baidu.com";
	private static final String baseUrlforName = "https://baike.baidu.com/item/"; 
		
	public static void main(String[] args) 
			throws ClientProtocolException, IOException
	{
//		BE VERY CAREFUL!

		String names = "李建国|温孚江|尹晋华|罗布江村|秦卫江|虞红鸣|陈代平|雒树刚|耿梅|鹿心社|易鹏飞|蒋超良|李秦生|朱学庆|刘小河|尔肯江·吐拉洪|饶南湖|陈建民|王清宪|李小鹏|黄世勇|李堂堂|刘杰|刘保威|毛生武|魏增军|甲热·洛桑丹增|杜黎明|孙永春|陈进玉|郭金龙|安焕晓|张宝顺|周先旺|高枫|解维俊|王增力|李国梁|肖建春|张家明|任正晓|马伟|田成江|张力|于莎燕|成其圣|时光辉|胡志强|陈润儿|王三运|张复明|赖晓岚|龙超云|马文云|招玉芳|刘上洋|程连元|龚建华|李洪义|唐坚|姜志刚|朱民|丁小强|田纪云|范长龙|周忠轩|杜学军|欧阳斌|铁力瓦尔迪·阿不都热西提|孙效东|辛国斌|陈雍|房俐|黄小祥|甲热·洛桑丹增|周雅光|姜杰|张鸿铭|张国清|郑亚军|鲁俊|徐钢|陈绿平|李湘林|赵铭|李康|庄如顺|李群|许立全|张启生|韩胜球|尚福林|曹卫星|郭安|龚毅|习近平|李斌|刘玉顺|程晓阳|沈宝昌|马春雷|万玛多杰|李恭进|侯晓春|侍俊|郭启俊|栗震亚|李登菊|罗凉清|赵奇|王炯|程红|李鸿忠|黄政红|林木声|冮瑞|王少玄|尹建业|詹夏来|黄伟京|黄强|杨松|吕维峰|黄楚平|姜帆|张庆岩|邓小刚|王晨|贺一诚|杨晓渡|班程农|车光铁|董卫民|朱民阳|张茂才|盛茂林|黄日波|张璞|郭洪昌|刘家升|赵祝平|孙建平|彭佩云|黄玮|黄跃进|李修松|徐力群|江涛|杨兴平|廉素|杜昌文|李公乐|蒋斌|郑广富|刘志强|王玉明|李柏拴|杨安娣|王赋|葛慧君|张广智";
//		String names = "姜樑";
		String[] nameList = names.split("\\|");
		Arrays.stream(nameList)
			.distinct()
			.forEach(name -> {
				try {
					strategySelect(name);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			});
		// handle the links of a repetitive name saved by secondStrategy()
//		handleLinks();

		
		
//		String leaderBasicPath = "E:/data/b.csv";
//		String leaderExpPath = "E:/data/e.csv";
		
		String leaderBasicPath = "E:/data/leaderBasic12.csv";
		String leaderExpPath = "E:/data/leaderExp12.csv";
		CsvWriter csvLeaderBasic = new CsvWriter(leaderBasicPath, ',', Charset.forName("gbk"));
		CsvWriter csvLeaderExp = new CsvWriter(leaderExpPath, ',', Charset.forName("gbk"));
		
		String[] basicHead = {"姓名","性别","民族","出生年月","籍贯",
				"入党时间","参加工作时间","教育背景","最高学历","职务"};
		String[] expHead = {"姓名","职务","起始时间","终止时间","任职信息"};
		
		csvLeaderBasic.writeRecord(basicHead);
		csvLeaderExp.writeRecord(expHead);
		leadersList.stream().forEach(leader->{
			writeBasic(csvLeaderBasic, leader);
			writeExp(csvLeaderExp, leader);
			print("write leader: "+ leader.name);
		});
		csvLeaderBasic.close();
		csvLeaderExp.close();
		print("Done!");
		
//		
		String repetitiveNamePath = "E:/data/repetitiveNames12.csv";
		CsvWriter csvrepetitiveName = new CsvWriter(repetitiveNamePath, ',', Charset.forName("gbk"));
		repetitiveName.stream().forEach(name->{
			try {
				csvrepetitiveName.write(name);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});
		csvrepetitiveName.close();
//		
		
	}
	
	/**
	 * core function: combine all other components.
	 * @param name: name of the leader
	 * @throws ClientProtocolException
	 * @throws IOException
	 */
	private static void strategySelect(String name) 
			throws ClientProtocolException, IOException
	{
		Document doc = getHtml(baseUrlforName, name);
		String judgeResult = judgeByDoc(doc);
		if (judgeResult == "Empty")
			print("NOT found" + name);
		else if (judgeResult == "One")
			firstStrategy(name, doc);
		else
			repetitiveName.add(name);
//			secondStrategy(name, doc);
	}
	
	
	/**
	 * core function: commit the first strategy --
	 * save the Leader to global parameter leadersList
	 * @param name: name of the leader
	 * @param doc: doc of the leader
	 */
	private static void firstStrategy(String name, Document doc)
	{
		Elements para = doc.select("div.para[label-module='para']");
		List<String> cleanedPara = cleanPara(para, name); 
		List<String[]> experience = parseDateAndEvent(cleanedPara);
		Leader leader = new Leader(name, experience);
		setBasicInfo(leader, doc);
		leadersList.add(leader);
	}
	
	/**
	 * core function: commit the second strategy -- 
	 * save the links to global parameter linksList 
	 * and save or discard the current doc 
	 * @param name: name of a leader
	 * @param doc: doc of a leader
	 */
	private static void secondStrategy(String name, Document doc)
	{
		// name is repetitive, so save the links firstly
		selectLinks(name, doc);
		
		String description = doc.select("span.selected").text();
		String experience = doc.select("h2").text();
		if (judgeByDescription(description)){
			if(judgeByh2(experience))
				firstStrategy(name, doc);
			else
				print(name + " :failed h2 judgement");
		}
		else
			print(name + " :failed description judgement");
	}
	
	/**
	 * core function: select the links of a repetitive name using
	 * its social description 
	 * @param name
	 * @param doc
	 */
	private static void selectLinks(String name, Document doc)
	{
		Elements ul = doc.select("ul.polysemantList-wrapper");
		Elements links = ul.select("li.item");
		links.stream()
			.filter(rawLink -> judgeByDescription(rawLink.text()) == true)
			.forEach(rawLink-> {
				Elements aLink = rawLink.select("a");
				String link = aLink.attr("href");
				linksList.add(link);
			});
	}
	
	
	/**
	 * core function: check the global parameter linksList 
	 * and save the proper Leader object
	 * @throws ClientProtocolException
	 * @throws IOException
	 */
	private static void handleLinks() 
			throws ClientProtocolException, IOException
	{
		if (!linksList.isEmpty()){
			for (String link: linksList)
			{
				try {
					Document doc = getHtml(baseUrlforLink, link);
					String h2 = doc.select("h2").text();
					if (judgeByh2(h2)){
						String name = getNameByDoc(doc);
						firstStrategy(name, doc);
					} else
						print(link + " :failed h2 judgement");
				} catch (UnknownHostException e) {
					print(link + " :bad link!");
				};
			}
		}
	}
	
	/**
	 * function: extract the raw html of given leader
	 * @param name: name of the leader
	 * @return: raw html of the leader in baike.baidu.com
	 * @throws ClientProtocolException
	 * @throws IOException
	 */
	public static Document getHtml(String base, String name) 
			throws ClientProtocolException, IOException
	{
		CloseableHttpClient httpclient = HttpClients.createDefault();
		HttpGet httpget = new HttpGet(base + name);
		CloseableHttpResponse response = httpclient.execute(httpget);
		HttpEntity entity = response.getEntity();
		String html = EntityUtils.toString(entity , "utf-8").trim();
		Document doc = Jsoup.parse(html);
		return doc;
	}
	

	/**
	 * function: clean the target paragraph
	 * @param para: initial dirty para
	 * @param name: name of the leader
	 * @return a cleaner paragraph
	 */
	private static List<String> cleanPara(Elements para, String name)
	{
		Predicate<String> startsWithYear = s -> s.matches("\\d{4}..*");
		Predicate<String> endsWithDay = s -> !s.matches("\\d{4}年\\d{1,2}月\\d{1,2}日..*");
		List<String> cleanedPara = para.stream()
				.map(Element::text)
				.map(s -> s.replaceAll("^.*\\(\\d+张\\) |^"+name+" ", ""))
				.map(s -> s.replaceAll("\\[\\d+(\\-\\d+)?\\]|；", ""))
				.filter(startsWithYear)
				.filter(endsWithDay)
				.collect(Collectors.toList());
		return cleanedPara;
	}
	
	
	/**
	 * function: return leader's experience
	 * @param cleanedPara: return by cleanPara()
	 * @return experience: List of date2AndEvent(start, end, event)
	 */
	private static List<String[]> parseDateAndEvent(List<String> cleanedPara)
	{
		Pattern pattern = Pattern.compile("^[\\d\\.\\-━—－~― 至今年月日起后到]+，?");
		List<String[]> experience = cleanedPara.stream()
				.map(s -> parseHelper(pattern, s))
				.map(s -> cleanHelper(s))
				.map(s -> splitAndReplaceHelper(s))
				.collect(Collectors.toList());
		return experience;
	}
	
	
	/**
	 * helper function ONLY called by parseDateAndEvent()
	 * @param dateAndEvent
	 * @return an clearer dateAndEvent 
	 */
	private static String[] cleanHelper(String[] dateAndEvent)
	{
		String pat = "[ ，后起]";
		dateAndEvent[0] = dateAndEvent[0].replaceAll(pat, "");
		return dateAndEvent;
	}
	
	
	/**
	 * helper function ONLY called by parseDateAndEvent()
	 * split the date to something like (start, end)
	 * @param dateAndEvent: (date, event) 
	 * @return date2AndEvent: (startDate, endDate, event)
	 */
	private static String[] splitAndReplaceHelper(String[] dateAndEvent)
	{
		String pat = "[\\-━—－―~ 至到]+";
		String[] date2AndEvent = new String[3];
		String[] date2 = dateAndEvent[0].split(pat);
		if (date2.length == 1){
			date2AndEvent[0] = date2[0].replaceAll("[年 ]", "\\.").replaceAll("月", "");
			date2AndEvent[1] = "";
		} else {
			date2AndEvent[0] = date2[0].replaceAll("[年 ]", "\\.").replaceAll("月", "");
			date2AndEvent[1] = date2[1].replaceAll("[年 ]", "\\.").replaceAll("月|今", "");
		}
		date2AndEvent[2] = dateAndEvent[1];			
		return date2AndEvent;
	}
	
	
	/**
	 * helper function ONLY called by parseDateAndEvent()
	 * @param pattern: regex matched the date, defined in parseDateAndEvent()
	 * @param string:  an event record with date
	 * @return arrays consists of date and event
	 */
	private static String[] parseHelper(Pattern pattern, String string)
	{
		Matcher matcher = pattern.matcher(string);
		matcher.find();
		int end = matcher.end();
		String date = string.substring(0, end);
		String event = string.substring(end);
		String[] dateAndEvent= new String[2];
		dateAndEvent[0] = date; 
		dateAndEvent[1] = event;
		return dateAndEvent;
	}

	
	/**
	 * Utility function: get the name of leader by document
	 * @param doc: doc of a leader
	 * @return: name of the leader
	 */
	private static String getNameByDoc(Document doc)
	{
		String name = doc.select("h1").text();
		return name;
	}
	
	
	/**
	 * Utility function: judge whether a name is repetitive 
	 * according to its doc.
	 * @param doc: doc of a leader
	 * @return "Empty" means not found
	 * "More" means the name is repetitive
	 * "One" means the name is unique
	 */
	private static String judgeByDoc(Document doc)
	{
		Elements sorry = doc.select("p.sorryCont");
		String text = sorry.text();
		if (!text.isEmpty()){
			return "Empty";
		}
		Elements links = doc.select("ul.polysemantList-wrapper");
		if (!links.isEmpty()){
			return "More";
		} else
			return "One";
	}
	
	
	/**
	 * Utility function: JuST Debug! judge whether a name is repetitive
	 * @param name: name of leader
	 * @throws ClientProtocolException
	 * @throws IOException
	 */
	public static void judgeByName(String name) 
			throws ClientProtocolException, IOException
	{
		Document doc = getHtml(baseUrlforName, name);
		Elements sorry = doc.select("p.sorryCont");
		String text = sorry.text();
		if (!text.isEmpty()){
			print(text);
			return;
		}
		Elements links = doc.select("ul.polysemantList-wrapper");
		if (!links.isEmpty()){
			print(links);
		} else
			print("Single!");
	}
	
}