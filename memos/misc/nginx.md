# Nginx 跨域
location / {
    proxy_pass http://www.baidu.com;
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods "POST, GET, OPTIONS";
    add_header Access-Control-Allow-Headers "Origin, Authorization, Accept";
    add_header Access-Control-Allow-Credentials true;
}
