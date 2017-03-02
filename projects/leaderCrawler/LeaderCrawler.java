package leaderCrawler;

// the same package
import static leaderCrawler.LeaderUtinity.*;
import static leaderCrawler.LeaderInfoCrawler.*;

// java package
import java.io.IOException;
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

/**
 * @author Mory
 */
public class LeaderCrawler
{
	private static ArrayList<String> linksList = new ArrayList<String>();
	private static ArrayList<Leader> leadersList = new ArrayList<Leader>();
	private static final String baseUrlforLink = "http://baike.baidu.com";
	private static final String baseUrlforName = "https://baike.baidu.com/item/"; 
		
	public static void main(String[] args) 
			throws ClientProtocolException, IOException
	{
	
		String names = "路飞";
		String[] nameList = names.split("\\|");
		Arrays.stream(nameList)
			.distinct()
			.forEach(name -> {
				try {
					Document doc = getHtml(baseUrlforName, name);
					String summary = getSummaryByDoc(doc);
					testSummary(summary);
					print("");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			});
//		// handle the links of a repetitive name saved by secondStrategy()
//		handleLinks();
//		// debug 
//		for (Leader l: leadersList)
//		{
//			l.showExperience();
//			print("============================================\n");
//		}
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
			print("NOT found");
		else if (judgeResult == "One")
			firstStrategy(name, doc);
		else
			secondStrategy(name, doc);		
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
				Document doc = getHtml(baseUrlforLink, link);
				String h2 = doc.select("h2").text();
				if (judgeByh2(h2)){
					String name = getNameByDoc(doc);
					firstStrategy(name, doc);
				} else
					print(link + " :failed h2 judgement");
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
		List<String> cleanedPara = para.stream()
				.map(Element::text)
				.map(s -> s.replaceAll("^.*\\(\\d+张\\) |^"+name+" ", ""))
				.map(s -> s.replaceAll("\\[\\d+(\\-\\d+)?\\]", ""))
				.filter(startsWithYear)
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
		Pattern pattern = Pattern.compile("^[\\d\\.\\-━—－ 至今年月日起后到]+，?");
		List<String[]> experience = cleanedPara.stream()
				.map(s -> parseHelper(pattern, s))
				.map(s -> cleanHelper(s))
				.map(s -> splitHelper(s))
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
	private static String[] splitHelper(String[] dateAndEvent)
	{
		String pat = "[\\-━—－ 至到]+";
		String[] date2AndEvent = new String[3];
		String[] date2 = dateAndEvent[0].split(pat);
		if (date2.length == 1){
			date2AndEvent[0] = date2[0];
			date2AndEvent[1] = "";
		} else {
			date2AndEvent[0] = date2[0];
			date2AndEvent[1] = date2[1];
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