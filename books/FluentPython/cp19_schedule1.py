# cp19_schedule1.py

import warnings
import cp19_dynamic_attributes_properties

DB_NAME = 'data/schedule1_db'
CONFERENCE = 'conference.115'

class Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_db(db):
    raw_data = cp19_dynamic_attributes_properties.load()
    warnings.warn('loading ' + DB_NAME)
    for collection, rec_list in raw_data['Schedule'].items():
        record_type = collection[:-1]
        for record in rec_list:
            key = '{}.{}'.format(record_type, record['serial'])
            record['serial'] = key
            db[key] = Record(**record)



if __name__ == '__main__':
    import shelve

    # open an empty shelf
    db = shelve.open(DB_NAME)
    if CONFERENCE not in db: # True
        load_db(db)
    speaker = db['speaker.3471']
    type(speaker)
    print(speaker.name, speaker.twitter)