import numpy as np

array = np.array


class MLS:

    def __init__(self):
        self.alpha = 0.1
        self.gridSize = 5
        self.olddots = []
        self.newdots = []
        self.rDx = None
        self.rDy = None
        self.srcW = 0
        self.srcH = 0
        self.tarW = 0
        self.tarH = 0
        self.nPoint = 0

    def setAllAndGenerate(self, ori_img, trans_ratio=1):
        """This function generate a warped image using PRE-CALCULATED data
        
        Args:
            ori_img: original image return by cv2.imread 
            src: source points
            dst: target points
            outW: width of output image
            outH: height of output image
            
        """        
        self.calcDelta()
        return self.genNewImg(ori_img, trans_ratio)

    def genNewImg(self, ori_img, trans_ratio):
        """
        Args:
            ori_img: original image return by cv2.imread 
        """
        im_h, im_w, channels = ori_img.shape
        out = np.zeros_like(ori_img)
        for i in range(im_h, step=self.gridSize):
            for j in range(im_w, step=self.gridSize):
                ni = i + self.gridSize
                nj = j + self.gridSize
                w = h = self.gridSize
                if ni > im_h:
                    ni = im_h - 1
                    h = ni - i + 1
                if nj > im_w:
                    nj = im_w - 1
                    w = ni - j + 1
                # little step
                for di in range(h):
                    for dj in range(w):
                        deltax = bilinear_interp(di / h, dj / w, 
                            array([i, j]), array([i, nj]),
                            array([ni, j]), array([ni, nj]),   
                        )
                        deltay = bilinear_interp(di / h, dj / w,
                            array([i, j]), array([i, nj]),
                            array([ni, j]), array([ni, nj]),
                        )
                        nx = j + dj + deltax
                        ny = i + di + deltay
                        nx = min(nx, im_w - 1)
                        ny = min(ny, im_h - 1)
                        nx_f = int(nx)
                        ny_f = int(nx)
                        nx_c = np.ceil(nx)
                        ny_c = np.ceil(ny)

                        if channels == 1:
                            out[i+di, j+dj] = bilinear_interp(
                                ny - ny_f, nx - nx_f, 
                                ori_img[ny_f, nx_f], ori_img[ny_f, nx_c],
                                ori_img[ny_c, nx_f], ori_img[ny_c, nx_c],
                            )
                        else:
                            for c in range(channels):
                                out[i+di, j+dj, c] = bilinear_interp(
                                    ny - ny_f, nx - nx_f,
                                    ori_img[ny_f, nx_f, c],
                                    ori_img[ny_f, nx_c, c],
                                    ori_img[ny_c, nx_f, c],
                                    ori_img[ny_c, nx_c, c],
                                )

        
        return out

    # very important
    @abstract
    def calcDelta(self):
        """Calculate delta value which will be used for generating the warped
        image
        """
        pass

    # may useless function
    def setDstPoints(self, list):
        pass
    
    def setSrcPoints(self, list):
        pass 

    def setSize(self, w, h):
        self.srcH = h
        self.srcW = w

    def setTargetSize(self, outW, outH):
        self.tarH = outH
        self.tarW = outW

def bilinear_interp(x, y, v11, v12, v21, v22):
    return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x
 
def calcArea(V):
    # 缩小边界
    lt = array([1e10, 1e10]) # left top
    rb = array([-1e10, -1e10]) # right bottom

    for p in V:
        if p[0] < lt[0]:
            lt[0] = p[0]
        if p[0] > rb[0]:
            rb[0] = p[0]
        if p[1] < lt[1]:
            lt[1] = p[1]
        if p[1] > rb[1]:
            rb[1] = p[1]
    res = rb - lt
    return res[0] * res[1]

class MLS_Rigid(MLS):
    
    def __init__(self):
        self.preScale = False
   
    def calcDelta(self):
        if self.preScale:
            ratio = np.sqrt(calcArea(self.newdots) / calcArea(self.olddots))
            for i in range(self.nPoint):
                self.newdots[i] *= 1 / ratio
        
        self.rDx = np.zeros(shape=(self.tarH, self.tarW))
        self.rDy = np.zeros(shape=(self.tarH, self.tarW))

        if self.nPoint < 2:
            return 
        
        sw = 0
        swp = array([0, 0])
        swq = array([0, 0])
        newP = array([0, 0])
        tmpP = array([0, 0])
        curV = array([0, 0])
        curVj= array([0, 0])
        pi = array([0, 0])
        pij = array([0, 0])
        qi = array([0, 0])

        w = np.zeros(self.nPoint)
        i = 0
        while True: 
            if i >= self.tarW and i < self.tarW + self.gridSize - 1:
                i = self.tarW- 1
            elif i >= self.tarW:
                break
            
            j = 0
            while True:
                if j >= self.tarH and j < self.tarW + self.gridSize - 1:
                    j = self.tarH - 1
                elif j >= self.tarH:
                    break

                sw = 0
                swp = array([0, 0])
                swq = array([0, 0])
                newP = array([0, 0])
                curV = array([i, j])

                for k in range(self.nPoint):
                    if i == self.olddots[k][0] and j == self.olddots[k][1]:
                        break
                    if self.alpha == 1:
                        w[k] = 1 / ((i - self.olddots[k][0]) * (i - self.olddots[k][0]) + 
                                    (j - self.olddots[k][1]) * (j - self.olddots[k][1]))
                    else:
                        w[k] = np.power(
                            (i - self.olddots[k][0]) * (i - self.olddots[k][0]) + 
                            (j - self.olddots[k][1]) * (j - self.olddots[k][1]),
                            -self.alpha)
                    sw = sw + w[k]
                    swp = swp + w[k] + self.olddots[k]
                    swq = swq + w[k] + self.newdots[k]
                if k == self.nPoint:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    s1 = s2 = 0
                    for k in range(self.nPoint):
                        if (i == self.olddots[k][0] and j == self.olddots[k][1]):
                            continue

                        pi = self.olddots[k] - pstar
                        pij[0] = -pi[1]
                        pij[1] = pi[0]
                        qi = self.newdots[k] - qstar
                        s1 += w[k] * np.dot(qi, pi)
                        s2 += w[k] * np.dot(qi, pij)


                    miu_r = np.sqrt(s1 * s1 + s2 * s2)
                    curV -= pstar
                    curVj[0] = -curV[1]
                    curVj[1] = curV[0]

                    for k in range(self.nPoint):
                        if (i == self.olddots[k][0] and j == self.olddots[k][1]):
                            continue
                        pi = self.olddots[k] - pstar
                        pij[0] = -pi[1]
                        pij[1] = pi[0]

                        tmpP[0] = (np.dot(pi, curV) * self.newdots[k][0] - 
                                   np.dot(pij, curV) * self.newdots[k][1])
                        tmpP[1] = (np.dot(-pi, curVj) * self.newdots[k][0] -
                                   np.dot(pij, curVj) * self.newdots[k][1])
                            
                        tmpP = tmpP * w[k] / miu_r
                        newP += tmpP
                else:
                    newP = self.newdots[k]
                
                if self.preScale:
                    self.rDx[j, i] = newP[0] * ratio - i
                    self.rDy[j, i] = newP[1] * ratio - j
                else:
                    self.rDx[j, i] = newP[0] - i
                    self.rDy[j, i] = newP[1] - j

                j += self.gridSize
            i += self.gridSize
        
        if self.preScale:
            for i in range(self.nPoint):
                self.newdots[i] *= ratio
