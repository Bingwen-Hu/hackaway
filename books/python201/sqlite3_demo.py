import sqlite3


def create_database(database):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    create_sqls = """
    CREATE TABLE albums
    (title text, artist text, release_date text, 
    publisher text, media_type text, ind int32)
    """
    
    cursor.execute(create_sqls)

    insert_sqls = """
    INSERT INTO albums VALUES 
    ('Glow', 'Andy Hunter', '7/25/2012', 'Xplore', 'MP3', 1)
    """
    cursor.execute(insert_sqls)
    conn.commit()

    multiinsert = """INSERT INTO albums VALUES (?, ?, ?, ?, ?, ?)"""
    values = [
        ('Glow', 'Andy Hunter', '7/25/2012', 'Xplore', 'MP3', 2),
        ('Glow', 'Andy Hunter', '7/25/2012', 'Xplore', 'MP3', 3),
    ]
    cursor.executemany(multiinsert, values)
    conn.commit()
    cursor.close()
    conn.close()

def select_all_albums():
    conn = sqlite3.connect('local.db')
    cursor = conn.cursor()

    sqls = "SELECT * FROM albums"
    cursor.execute(sqls)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

if __name__ == '__main__':
    create_database('local.db')
    result = select_all_albums()
    for row in result:
        rowstr = [str(r) for r in row]
        print(' '.join(rowstr))

    import os
    os.remove('local.db')