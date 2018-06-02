import matplotlib.pyplot as plt
import numpy as np

# --------------------- arcsin x ----------------------- 
# y = arcsin(x), D = [-1, 1], simple increase, V = [-pi/2, pi/2]
x = list(range(-10000, 10000, 1))
x = np.array(x)/10000
y = np.arcsin(x)
print(np.arcsin(1))
plt.subplot(231)
plt.title('y = arcsin x')
plt.plot(x, y)


# --------------------- arccos x -----------------------
# y = arccos(x) D = [-1, 1], simple descrease, V = [0, pi]
x = list(range(-10000, 10000, 1))
x = np.array(x)/10000
y = np.arccos(x)
print(np.arccos(-1))
plt.subplot(232)
plt.title('y = arccos x')
plt.plot(x, y)



# --------------------- arctan x -----------------------
# y = arccos(x) D = (-oo, +oo), simple descrease, V = [-pi/2, pi/2]
x = list(range(-10000, 10000, 1))
x = np.array(x)/1000
y = np.arctan(x)
print(np.arctan(10000000000))
plt.subplot(233)
plt.title('y = arctan x')
plt.plot(x, y)



# ----------------------- tan x ------------------------
# y = tan(x) D = (-oo, +oo) x-> pi/2 y->+oo, x->-pi/2 y->-oo
x = list(range(-157, 158, 1))
x = np.array(x)/100
y = np.tan(x)
plt.subplot(236)
plt.title('y = tan x')
plt.plot(x, y)



# ----------------------- tan x ------------------------
# y = tan(x) D = (-oo, +oo) x-> pi/2 y->+oo, x->-pi/2 y->-oo
x = list(range(-157, 158, 1))
x = np.array(x)/100
y = x / (1 + np.power(x, 2))
plt.subplot(234)
plt.title('y = x / (1 + x^2)')
plt.plot(x, y)
plt.show()
