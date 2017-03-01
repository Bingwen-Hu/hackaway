package leaderCrawler;

import java.util.List;

public class Leader
{
	public String name;
	public List<String[]> experience;
	public Leader(String name, List<String[]> experience)
	{
		this.name = name;
		this.experience = experience;
	}
	
	public void showExperience()
	{
		System.out.println(name);
		for (String[] s : experience)
		{
			System.out.printf("%s|%s|%s%n", s[0], s[1], s[2]);
		}
	}
	
	@Override 
	public String toString()
	{
		return String.format("name: %-8s%nexperience: %d in all%n", 
				name, experience.size());
	}
	
}