import psp

img = 'imgs/000008_0.jpg'
prediction = psp.parse(img)
psp.save(prediction, "result.png")