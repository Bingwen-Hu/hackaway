package leaderCrawler;

import static leaderCrawler.LeaderUtinity.*;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

public class LeaderInfoCrawler
{
	// 性别
	private static String gender = "(?<=，|。| )(男|女)(?=，|。| )";
	
	// 民族：只能匹配单字
	private static String race = "([\\u0391-\\uFFE5]族)";
	
	// 出生于xxxx， xxxx出生
	private static String birthDate = "(([年月\\d]+)(?=出?生))|(?<=出?生于)([年月\\d]+)";
	
	// 省级行政单位的首字
	private static String 
		nativePlace = "(?<=[，。])([北上天重黑辽吉河湖山陕安浙江福广海四云贵青甘台内宁新西香澳]\\S+?人)";
	
	// 1974年1月(加)入((中国)共产)党
	private static String joinPartyDate = "([年月\\d]+)(?=加?入中?国?共?产?党)";
	
	// 1969年1月(加入中国共产党并)参加(革命)工作
	private static String joinWorkDate = "([年月\\d]+)+(?=([\\u0391-\\uFFE5]+)?参加(革命)?工作)";
	
	// 清华大学人文社会学院马克思主义理论与思想政治教育专业毕业，在职研究生学历，法学博士学位。
	private static String eduBackground = "(?<=[，。])([\\u0391-\\uFFE5]*?(专业|党校|学历))"
			+ "((\\S+)?[博士|研究生|硕士|博士后])?";
	
	// 现任中国共产党中央委员会总书记
	private static String position = "(现任|曾任)[\\u0391-\\uFFE5]+";    
	
	
	
	public static void testSummary(String summary)
	{
		print("name:           "+getName(summary));
		print("race:           "+getBasic(summary, race));
		print("eduBackground:  "+getBasic(summary, eduBackground));
		print("birth date:     "+getBasic(summary, birthDate));
		print("Join work Date: "+getBasic(summary, joinWorkDate));
		print("Gender:         "+getBasic(summary, gender));
		print("Join Party Date:"+getBasic(summary, joinPartyDate));
		print("Position:       "+getBasic(summary, position));
		print("native place:   "+getBasic(summary, nativePlace));
		print("Edu Level:      "+getEduLevel(summary));
		
	}
	
	
	public static void setBasicInfo(Leader leader, Document doc)
	{
		String summary = getSummaryByDoc(doc);
		leader.birthDate = getBasic(summary, birthDate);
		leader.eduBackground = getBasic(summary, eduBackground);
		leader.eduLevel = getEduLevel(summary);
		leader.gender = getBasic(summary, gender);
		leader.joinPartyDate = getBasic(summary, joinPartyDate);
		leader.joinWorkDate = getBasic(summary, joinWorkDate);
		leader.nativePlace = getBasic(summary, nativePlace);
		leader.position = getBasic(summary, position);
		leader.race = getBasic(summary, race);
	}
	
	public static String getSummaryByDoc(Document doc)
	{
		String summary = doc.select("div.lemma-summary").text();
		if (summary.isEmpty()) {
			Elements para = doc.select("div.para[label-module='para']");
			summary = para.first().text();
		}
		return summary;
	}
	
	// buggy!
	public static String getName(String summary)
	{
		String name = summary.split("，")[0];
		return name;
	}
	
	public static String getBasic(String summary, String regex)
	{
		Pattern pattern = Pattern.compile(regex);  
		Matcher matcher = pattern.matcher(summary);  
		Boolean exist = matcher.find();
		if (exist)
			return matcher.group();
		return "";
	}
	
	public static String getEduLevel(String summary)
	{
		String eduLevel = "";
		if (summary.contains("博士后"))
			eduLevel = "博士后";
		else if (summary.contains("博士"))
			eduLevel = "博士";
		else if (summary.contains("研究生") || summary.contains("硕士"))
			eduLevel = "硕士";
		else if (summary.contains("大学"))
			eduLevel = "本科";
		else if (summary.contains("大专") || summary.contains("学院"))
			eduLevel = "大专";
		else if (summary.contains("高中"))
			eduLevel = "高中";
		else if (summary.contains("初中"))
			eduLevel = "初中";
		else
			eduLevel = "";
		return eduLevel;
	}
	
}