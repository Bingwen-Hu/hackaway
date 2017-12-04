# fluent python hard way to finish but I will continue!

# A user-defined class implementing __getattr__ can implement
# "virtual attributes" by computing values on the fly whenever
# somebody tries to read a nonexistent attribute like obj.noexist


#JSON Format
#{ "Schedule":
#    { "conferences": [{"serial": 115 }],
#      "events": [
#        { "serial": 34505,
#          "name": "Why Schools Don´t Use Open Source to Teach Programming",
#          "event_type": "40-minute conference session",
#          "time_start": "2014-07-23 11:30:00",
#          "time_stop": "2014-07-23 12:10:00",
#          "venue_serial": 1462,
#          "description": "Aside from the fact that high school programming...",
#          "website_url": "http://oscon.com/oscon2014/public/schedule/detail/34505",
#          "speakers": [157509],
#          "categories": ["Education"] }
#        ],
#      "speakers": [
#          { "serial": 157509,
#          "name": "Robert Lefkowitz",
#          "photo": null,
#          "url": "http://sharewave.com/",
#          "position": "CTO",
#          "affiliation": "Sharewave",
#          "twitter": "sharewaveteam",
#          "bio": "Robert ´r0ml´ Lefkowitz is the CTO at Sharewave, a startup..." }
#          ],
#      "venues": [
#          { "serial": 1462,
#          "name": "F151",
#          "category": "Conference Venues" }
#          ]
#    }
#}




from urllib.request import urlopen
import warnings
import os
import json
URL = 'http://www.oreilly.com/pub/sc/osconfeed'
JSON='data/osconfeed.json'
def load():
    if not os.path.exists(JSON):
        msg = 'downloading {} to {}'.format(URL, JSON)
        warnings.warn(msg)
        with urlopen(URL, encoding='utf-8') as remote, open(JSON, 'wb') as local:
            local.write(remote.read())
    with open(JSON, encoding='utf-8') as fp:
        return json.load(fp)


from collections import abc
import keyword
class FrozenJSON:
    """A read-only facade for navigating a JSON-like object
       using attribute notation
    """
    def __new__(cls, arg):
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping):
        self.__data = {}
        for key, value in mapping.items():
            if keyword.iskeyword(key):
                key += '_'
            self.__data[key] = value

    def __getattr__(self, name):
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return FrozenJSON(self.__data[name])

# Note: __new__ construct the object than if pass to __init__ if
# the object return by __new__ is the same as the one initialization.


# chapter 19 is an very important chapter about class.