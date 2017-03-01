package leaderCrawler;

public class LeaderUtinity
{
	
	public static boolean judgeByDescription(String description)
	{
		String[] descriptionList = {"ʡ", "��ί", "��ί", "���", "ʡ��", "�г�", "����"}; 
		boolean pass = false;
		for (String s : descriptionList){
			if(description.matches(".*"+s+".*")){
				pass = true;
				return pass;
			} 
		}
		return pass;
	}
	
	public static boolean judgeByh2(String h2)
	{
		boolean pass = false;
		if (h2.matches(".*����.*"))
			pass = true;
		return pass;
	}
	
	
	public static <T> void printArray(T[] array)
	{
		for (T element : array)
		{
			System.out.println(element);
		}
	}
	public static <T> void print(T element)
	{
		System.out.println(element);
	}
	public static void printf(String format, Object... args)
	{
		System.out.printf(format, args);
	}
}