package leaderCrawler;

import static leaderCrawler.LeaderUtinity.*;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

public class LeaderInfoCrawler
{
	// �Ա�
	private static String gender = "(?<=��|��| )(��|Ů)(?=��|��| )";
	
	// ���壺ֻ��ƥ�䵥��
	private static String race = "([\\u0391-\\uFFE5]��)";
	
	// ������xxxx�� xxxx����
	private static String birthDate = "(([����\\d]+)(?=��?��))|(?<=��?����)([����\\d]+)";
	
	// ʡ��������λ������
	private static String 
		nativePlace = "(?<=[����])([�������غ��ɼ��Ӻ�ɽ�°��㽭���㺣���ƹ����̨�����������]\\S+?��)";
	
	// 1974��1��(��)��((�й�)����)��
	private static String joinPartyDate = "([����\\d]+)(?=��?����?��?��?��?��)";
	
	// 1969��1��(�����й���������)�μ�(����)����
	private static String joinWorkDate = "([����\\d]+)+(?=([\\u0391-\\uFFE5]+)?�μ�(����)?����)";
	
	// �廪��ѧ�������ѧԺ���˼����������˼�����ν���רҵ��ҵ����ְ�о���ѧ������ѧ��ʿѧλ��
	private static String eduBackground = "(?<=[����])([\\u0391-\\uFFE5]*?(רҵ|��У|ѧ��))"
			+ "((\\S+)?[��ʿ|�о���|˶ʿ|��ʿ��])?";
	
	// �����й�����������ίԱ�������
	private static String position = "(����|����)[\\u0391-\\uFFE5]+";    
	
	
	
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
		String name = summary.split("��")[0];
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
		if (summary.contains("��ʿ��"))
			eduLevel = "��ʿ��";
		else if (summary.contains("��ʿ"))
			eduLevel = "��ʿ";
		else if (summary.contains("�о���") || summary.contains("˶ʿ"))
			eduLevel = "˶ʿ";
		else if (summary.contains("��ѧ"))
			eduLevel = "����";
		else if (summary.contains("��ר") || summary.contains("ѧԺ"))
			eduLevel = "��ר";
		else if (summary.contains("����"))
			eduLevel = "����";
		else if (summary.contains("����"))
			eduLevel = "����";
		else
			eduLevel = "";
		return eduLevel;
	}
	
}