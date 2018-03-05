""" mixin class make a normal object get some extra power while keeping the initial class
code clean and managable
"""
import pymysql
import pprint

class DemonMixin(object):
    """very strange, all demons connect each other through 
    a database server!"""

    server = None
    database = None
    cursor = None
    
    def setup_db(self, host, username, password, port, database):
        if self.server is not None:
            return 
        params = {'host': host,
                  'user': username, 
                  'password': password, 
                  'port': port,
                  'database': database,
                  'charset': 'utf8'}
        self.server = pymysql.connect(**params)
        self.database = database
        


    def execute(self, sql, params=None):
        self.cursor = self.server.cursor()
        result = self.cursor.execute(sql, params)
        return result
        
        
    

class Surpasser:
    """a surpasser without any power"""
    def __init__(self, name, surpass):
        self.name = name
        self.surpass = surpass


class Demon(DemonMixin, Surpasser):
    pass


if __name__ == '__main__':
    mory = Demon('Mory', "Demon")
    mory.setup_db(host='localhost', username='mory', 
        password='mory2016', port=3306, database='datamodel')
    mory.execute("show tables;")
    results = mory.cursor.fetchall()
    pprint.pprint(results)
    mory.cursor.close()
    mory.server.close()