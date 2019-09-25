from PIL import Image


path = 'dark.png'

img = Image.open(path)

colors = [12 * color for color in range(20)]

img.putpalette(colors)
img.save('color.png')


