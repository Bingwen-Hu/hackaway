from wordcloud import WordCloud

with open('webUtil.py') as f:
    text = f.read()

wordcloud = WordCloud().generate(text)

wordcloud.to_image('test.png')