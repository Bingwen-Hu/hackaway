# xml generate and store
# steps:
# 1. create a document object
# 2. using document object to create a root object
# 3. add data into root object
# 4. write out document object

# generate an xml file like this:
"""
<?xml version="1.0"?>
<data author="mory">
    <id/>
    <name/>
    <url/>
    <title>some title</title>
    <content>some content</content>
    <author/>
    <from/>
    <time>when</time>
    <image>image</time>
</data>
"""

# basic
import xml.dom.minidom as dom

doc = dom.Document()
root = doc.createElement("data")
root.setAttribute('author', 'Mory')
doc.appendChild(root)

nodenames = ['id', 'name', 'url', 'title', 'content', 'author', 'from', 'time', 'image']
for nodename in nodenames:
    node = doc.createElement(nodename)
    if nodename in ['title', 'content', 'time', 'image']:
        text = doc.createTextNode("%s content" % nodename)
        node.appendChild(text)
    root.appendChild(node)

with open('root.xml', 'w') as f:
    doc.writexml(f, indent='\t', addindent='\t', newl='\n', encoding='utf-8')