# ChainMap chain maps and work as one.

from collections import ChainMap


# where is mory?
mory = 'Mory'
Anndi = {'Ann': 'Ann',
         'Maple': 'Maple'}

Snowground = {'Jenny': 'Jenny',
              'Yidou': 'Yidou'}

China = {'Stone': 'Tree',
         'Mory': 'Mory'}

# normally
if mory in Anndi:
    print(Anndi[mory], " in Anndi")
elif mory in Snowground:
    print(f'{Snowground[mory]} in Snowground')
elif mory in China:
    print(f'{China[mory]} in China')


all_places = ChainMap(Anndi, Snowground, China)
print(all_places[mory])