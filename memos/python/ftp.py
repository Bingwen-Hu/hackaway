import ftplib
import os.path

host = 'localhost'
username = 'mory'
password = '2Foralfv'

server = ftplib.FTP(host)
server.login(username, password)



def upload(server, filename, bufsize=1024):
    pwd = server.pwd()
    basename = os.path.basename(filename)
    ftp_path = os.path.join(pwd, f"ftp_{basename}")
    with open(filename, 'rb') as file_local:
        print(ftp_path)
        server.storbinary("STOR %s" % ftp_path, file_local, bufsize)

upload(server, "/home/mory/hackaway/memos/python/pip.md")
