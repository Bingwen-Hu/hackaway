import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.HttpVersion;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.Header;
import org.apache.http.HttpEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.message.BasicHttpResponse;
import org.apache.http.util.EntityUtils;

public class LeaderCrawler
{
	public static void main(String[] args) 
			throws ClientProtocolException, IOException, URISyntaxException
	{
		CloseableHttpClient httpclient = HttpClients.createDefault();
		HttpGet httpget = new HttpGet("https://baike.baidu.com/item/Ï°½üÆ½");
		CloseableHttpResponse response = httpclient.execute(httpget);
		
		System.out.println(response.getStatusLine());
		try {
			
			System.out.println(response.getStatusLine());
			/*
			HttpEntity entity = response.getEntity();
			
			if (entity != null) {
				System.out.println(EntityUtils.toString(entity));
				System.out.println(entity.getContentEncoding());
				String respContent = EntityUtils.toString(entity , "utf-8").trim();
				System.out.println(respContent);
//				System.out.println(respContent.substring(0,1));
			}*/
		} finally {
		 response.close();
		}
	}

	public static void testFunction() throws IOException
	{
		StringEntity myEntity = new StringEntity("important message",
				 ContentType.create("text/plain", "UTF-8"));
				System.out.println(myEntity.getContentType());
				System.out.println(myEntity.getContentLength());
				System.out.println(EntityUtils.toString(myEntity));
				System.out.println(EntityUtils.toByteArray(myEntity).length);

	}
	
}