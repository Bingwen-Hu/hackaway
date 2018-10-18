import urllib.parse as urlparse


url = 'http://n.miaopai.com/media/wsq~EJAn9y3s5vuCqx4KO1MTYooe5UXw'
hostname = urlparse.urlparse(url).hostname
print(hostname)