import cv2
import numpy as np
import time


# BGR
init_color = (0, 122, 0)

colors = [
    (255, 255, 255), # white
    (100, 255, 100), # green
    (0, 255, 0),
    (255, 100, 100), # blue
    (255, 0, 0),
    (100, 100, 255), # red 
    (0, 0, 255),
    (0, 255, 255), # yellow
]
# draw a bar as experiment
bar = np.zeros((30, 300, 3), np.uint8)

def draw(bar, color):
    bar[..., :] = color
    return bar

def twinkle():
    global bar
    while True:
        for color in colors:
            bar = draw(bar, color)
            cv2.imshow('bar', bar)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
            time.sleep(0.5 / 15)


def gradual():
    global bar
    bar = draw(bar, init_color)
    for i in range(122):
        # bar[:, :, 0] -= 1
        bar[:, :, 1] -= 1
        bar[:, :, 2] += 2
        cv2.imshow('bar', bar)
        cv2.waitKey(1)
        time.sleep(0.5 / 15)
    cv2.destroyAllWindows()

twinkle()