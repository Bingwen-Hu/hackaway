package leaderCrawler;

import static leaderCrawler.LeaderUtinity.print;

import java.util.List;

public class Leader
{
	public String name;
	public String position;					// 职位
	public String gender;					// 姓别
	public String race;						// 民族
	public String birthDate;				// 出生年月
	public String joinPartyDate;			// 入党时间
	public String joinWorkDate;				// 参加工作时间
	public String nativePlace;				// 籍贯
	public String eduBackground;           	// 教育背景
	public String eduLevel;					// 教育水平:其它，初中，高中，本科，硕士，博士，博士后
	public List<String[]> experience;		// 履历
	
	// constructor: SHOULD be STRICTLY used
	public Leader(String name, List<String[]> experience)
	{
		this.name = name;
		this.experience = experience;
	}
	
	// for debug
	public void showExperience()
	{
		System.out.println(name);
		for (String[] s : experience)
		{
			System.out.printf("%s|%s|%s%n", s[0], s[1], s[2]);
		}
	}
	
	// for debug
	public void showBasicInfo()
	{
		print("name:           "+name);
		print("race:           "+race);
		print("eduBackground:  "+eduBackground);
		print("birth date:     "+birthDate);
		print("Join work Date: "+joinWorkDate);
		print("Gender:         "+gender);
		print("Join Party Date:"+joinPartyDate);
		print("Position:       "+position);
		print("native place:   "+nativePlace);
		print("Edu Level:      "+eduLevel);
	}
	
	
	@Override 
	public String toString()
	{
		return String.format("name: %-8s%nexperience: %d in all%n", 
				name, experience.size());
	}
	
}