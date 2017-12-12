import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.sql.Connection;
import java.sql.Date;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.List;

import com.csvreader.CsvReader;

public class DBUtil {
    
    private static final String URL="jdbc:mysql://192.168.1.224:3306/mysql";
    private static final String NAME="mory";
    private static final String PASSWORD="siriusdemon";
//    private static final String leaderExpPath = "E:/data/ALL_leaderExp2_14.csv";
//    private static final String leaderBasicPath = "E:/data/ALL_leaderBasic2_14.csv";
    
    public static void main(String[] args) throws Exception
    {
//    	Class.forName("com.mysql.jdbc.Driver");
//    	Connection conn = DriverManager.getConnection(URL, NAME, PASSWORD);
//		conn.setAutoCommit(false);
    	
    	CsvReader csvLeaderBasic = readCsv(leaderBasicPath);
		CsvReader csvLeaderExp = readCsv(leaderExpPath);

		String basicCommand = "insert into leader_basic_info (name, gender, race, birth_date, "
				+ "native_place, join_party, start_work, edu_background, degree, position) "
				+ "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
		PreparedStatement stmt = conn.prepareStatement(basicCommand, 
				Statement.RETURN_GENERATED_KEYS);
    	
		String expCommand = "Insert into leader_experience (leader_id, start_date, "
				+ "end_date, experience) values (?, ?, ?, ?) ";
		PreparedStatement pStatExp = conn.prepareStatement(expCommand);
		
		
		
    	try {
    		while (csvLeaderBasic.readRecord())
    		{	
    			writeBasic(stmt, csvLeaderBasic);
    			int leader_id = getLeaderID(stmt);
    			
    			while(csvLeaderExp.readRecord() && 
    					(csvLeaderExp.get("姓名").equals(csvLeaderBasic.get("姓名"))))
    			{
    				writeExp(pStatExp, csvLeaderBasic, csvLeaderExp, leader_id);
    				System.out.println("    experience....");
    			}
    			System.out.println(csvLeaderBasic.get("姓名") + " person finish.");
    		conn.commit();
    		}
    	} catch (Exception e) {
    		conn.rollback();
    		System.out.println("Roll Back!");
    		e.printStackTrace();
    	} finally {
    		conn.close();
    	}
    	System.out.println("Done!");
    }
    
    
    public static void writeBasic(PreparedStatement stmt, CsvReader csvLeaderBasic) 
    		throws IOException, SQLException, ParseException
    {
		stmt.setString(1, csvLeaderBasic.get("姓名"));
		stmt.setString(2, csvLeaderBasic.get("性别"));
		stmt.setString(3, csvLeaderBasic.get("民族"));
		stmt.setDate(4, strToDate(csvLeaderBasic.get("出生年月"), "basic"));
		stmt.setString(5, csvLeaderBasic.get("籍贯"));
		stmt.setDate(6, strToDate(csvLeaderBasic.get("入党时间"), "basic"));
		stmt.setDate(7, strToDate(csvLeaderBasic.get("参加工作时间"), "basic"));
		stmt.setString(8, csvLeaderBasic.get("教育背景"));
		stmt.setString(9, csvLeaderBasic.get("最高学历 "));
		stmt.setString(10, csvLeaderBasic.get("职务"));
		
		stmt.executeUpdate();
    }
    
    public static int getLeaderID(PreparedStatement stmt) throws SQLException
    {
    	ResultSet rs = stmt.getGeneratedKeys();
		int leader_id = 0;
		if (rs != null && rs.next())
			leader_id = rs.getInt(1);
		return leader_id;
    }
    
    public static void writeExp(PreparedStatement pStatExp, 
    		CsvReader csvLeaderBasic, CsvReader csvLeaderExp, int id) 
    				throws IOException, SQLException, ParseException
    {
    	pStatExp.setInt(1, id);
    	pStatExp.setDate(2, strToDate(csvLeaderExp.get("起始时间"), "exp")); 
    	pStatExp.setDate(3, strToDate(csvLeaderExp.get("终止时间"), "exp"));
    	pStatExp.setString(4, csvLeaderExp.get("任职信息"));

    	pStatExp.executeUpdate();
    }
    
    
    public static CsvReader readCsv(String filename) throws IOException
    {
    	CsvReader r = new CsvReader(filename, ',', Charset.forName("GBK"));
    	r.readHeaders();
    	return r;
    }
    
    public static Date strToDate(String strDate, String caller) throws ParseException
    {
    	String strFormat = "";
    	Boolean basicCalled = caller.equals("basic");
    	if (basicCalled)
    		strFormat = "yyyy/MM/dd";
    	else
    		strFormat = "yyyy.MM";
    	
    	// 如果空则补充
    	if (strDate.isEmpty()) {
    		if (basicCalled)
    			strDate = "9999/01/01";
    		else 
    			strDate = "9999.01";
    	}
    	if (strDate.length() == 4) {
    		if (basicCalled)
    			strDate = strDate + "/1/1";
    		else 
    			strDate = strDate + ".01";
    	}
    	
    	SimpleDateFormat format = new SimpleDateFormat(strFormat);
    	java.util.Date utilDate = format.parse(strDate);
    	Date sqlDate = new Date(utilDate.getTime());
    	return sqlDate;
    }

}
