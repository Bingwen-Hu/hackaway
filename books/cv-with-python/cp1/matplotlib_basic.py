from PIL import Image
import pylab
import numpy as np

def basic_draw():
    img = pylab.array(Image.open("E:/Mory/rose.jpg"))
    pylab.imshow(img)

    # some points
    x = [100, 100, 400, 400]
    y = [200, 500, 200, 500]

    # plot the points with red star-markers
    pylab.plot(x, y, 'r*')

    # line plot connecting the first two points
    pylab.plot(x[:2], y[:2], 'go-')
    pylab.plot(x[2:], y[2:], 'ks:')

    # horizon line
    pylab.plot((x[0], x[2]), (y[0], y[2]))

    # add title and show the plot
    pylab.title("Plotting: 'rose.jpg'")

    # to close the axis if you want
    pylab.axis('off')

    pylab.show()


def contour_and_hist():
    img = pylab.array(Image.open("E:/Mory/rose.jpg").convert("L"))
    print(type(img))
    # create a new figure
    pylab.figure()

    # don't use colors
    pylab.gray()

    # show contours with origin upper left corner
    pylab.contour(img, origin='image')
    pylab.axis('equal')
    pylab.axis('off')

    pylab.figure()
    pylab.hist(img.flatten(), 128)
    pylab.show()


def interactive():
    img = pylab.array(Image.open("E:/Mory/rose.jpg"))
    pylab.imshow(img)

    print("Please click 3 points")
    x = pylab.ginput(3)
    print("You had clicked: ", x)
    pylab.show()


def graplevel_transform():
    img = np.array(Image.open("E:/Mory/rose.jpg").convert('L'))

    img2 = 255 - img                # invert image
    img3 = (100.0/255) * img + 100  # clamp to interval 100...200
    img4 = 255.0 * (img/255.0) ** 2 # squared

    img3 = np.uint8(img3)
    img4 = np.uint8(img4)
    return img, img2, img3, img4