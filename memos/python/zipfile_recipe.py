# example for zipfile

import zipfile
import json

with zipfile.ZipFile('output.zip', mode='w') as zipwriter:
    zipwriter.write('root.xml') # just need a filepath
    zipwriter.write('toor.xml')
    comment = {"author": 'Mory',
                "date": 'you guess',
                'count': '2'}
    comment_str = json.dumps(comment)
    comment_bytes = comment_str.encode()
    zipwriter.comment = comment_bytes
