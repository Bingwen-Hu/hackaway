package leaderCrawler;

import static leaderCrawler.LeaderUtinity.print;

import java.util.List;

public class Leader
{
	public String name;
	public String position;					// ְλ
	public String gender;					// �ձ�
	public String race;						// ����
	public String birthDate;				// ��������
	public String joinPartyDate;			// �뵳ʱ��
	public String joinWorkDate;				// �μӹ���ʱ��
	public String nativePlace;				// ����
	public String eduBackground;           	// ��������
	public String eduLevel;					// ����ˮƽ:���������У����У����ƣ�˶ʿ����ʿ����ʿ��
	public List<String[]> experience;		// ����
	
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