package leaderCrawler;

import static leaderCrawler.LeaderUtinity.*;

import java.io.IOException;
import java.util.List;

import com.csvreader.CsvWriter;

public class LeaderOut
{
	public static void writeBasic(CsvWriter writer, Leader leader)
	{  
		try {
			String[] contents = {
					leader.name,
					leader.gender,
					leader.race,
					leader.birthDate,
					leader.nativePlace,
					leader.joinPartyDate,
					leader.joinWorkDate,
					leader.eduBackground,
					leader.eduLevel,
					leader.position};                      
			writer.writeRecord(contents);  
		} catch (IOException e) {  
			e.printStackTrace();  
		}  
	}
	
	public static void writeExp(CsvWriter writer, Leader leader)
	{  
		List<String[]> experience = leader.experience;
		experience.stream().forEach(exp -> {
			String[] contents = {
					leader.name,
					leader.position,
					exp[0],
					exp[1],
					exp[2],
			};
			try {
				writer.writeRecord(contents);
			} catch (IOException e) {
				print("go wrong! " + leader.name);
				e.printStackTrace();
			}
		});  
	}
	
}