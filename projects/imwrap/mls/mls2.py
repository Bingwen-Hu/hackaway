import numpy as np

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    

def bilinear_interp(x, y, v11, v12, v21, v22):
    """
    Args:
        x, y: 0 to 1 
        v11, v12, v21, v22: four points
    """
    return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x


def calcArea(V):
    """
    Args:
        V: list of Points 
    """
    lt = Point(1e10, 1e10)
    rb = Point(-1e10, -1e10)

    for i in V:
        lt.x = min(lt.x, i.x)
        rb.x = max(rb.x, i.x)
        lt.y = min(lt.y, i.y)
        rb.y = max(rb.y, i.y)
    return (rb.x - lt.x) * (rb.y - lt.y)


class MLS_Rigid(object):

    def __init__(self, preScale = False):
        self.oldDotL = []
        self.newDotL = []
        self.nPoint = 0
        self.rDx = np.array(0)
        self.rDy = np.array(0)
        self.srcW = None
        self.srcH = None
        self.tarW = None
        self.tarH = None
        self.gridSize = 5
        self.alpha = 1
        self.preScale = preScale

    
    def setTargetSize(self, outw, outH):
        self.tarW = outw
        self.tarH = outH
    
    def setSize(self, w, h):
        self.srcW = w
        self.srcH = h
    
    def setSrcPoints(self, qsrc):
        self.nPoint = len(qsrc)
        self.newDotL = qsrc

    def setDstPoints(self, qdst):
        self.nPoint = len(qdst)
        self.oldDotL = qdst

    def calcDelta(self):
        if self.preScale:
            ratio = np.sqrt(calcArea(self.newDotL) / calcArea(self.oldDotL))
            for i in range(self.nPoint):
                self.newDotL[i].x *= 1 / ratio
                self.newDotL[i].y *= 1 / ratio
        # init rDx rDy
        self.rDx = np.zeros([self.tarH, self.tarW])
        self.rDy = np.zeros([self.tarH, self.tarW])

        # omit situation
        if self.nPoint < 2:
            return
        
        # init
        sw = 0
        swq = Point(0, 0)
        qstar = Point(0, 0)
        newP = Point(0, 0)
        tmpP = Point(0, 0)
        swp = Point(0, 0)
        pstar = Point(0, 0)
        curV = Point(0, 0)
        curVJ = Point(0, 0)
        Pi = Point(0, 0)
        PiJ = Point(0, 0)
        Qi = Point(0, 0)
        miu_r = 0
        w = np.zeros(self.nPoint) # for points

        i = 0
        while True:
            # border check
            if i >= self.tarW and i < self.tarW + self.gridSize - 1:
                i = self.tarW - 1
            elif i >= self.tarW:
                break

            j = 0
            while True:
                if j >= self.tarH and j < self.tarH + self.gridSize - 1:
                    j = self.tarH - 1
                elif j >= self.tarH:
                    break

                curV.x = i
                curV.y = j
                for k in range(self.nPoint):
                    if i == self.oldDotL[k].x and j == self.oldDotL[k].y:
                        break
                    if self.alpha == 1:
                        w[k] = 1 / ((i - self.oldDotL[k].x)**2 + (j - self.oldDotL[k].y)**2)
                    else:
                        w[k] = np.power( (i - self.oldDotL[k].x)**2 + (j - self.oldDotL[k].y)**2, 
                                         -self.alpha)
                    sw = sw + w[k]
                    swp.x = swp.x + w[k] * self.oldDotL[k].x
                    swp.y = swp.y + w[k] * self.oldDotL[k].y
                    swq.x = swq.x + w[k] * self.newDotL[k].x
                    swq.y = swq.y + w[k] * self.newDotL[k].y

                if k == self.nPoint:
                    pstar.x = 1 / sw * swp.x
                    pstar.y = 1 / sw * swp.y
                    qstar.x = 1 / sw * swq.x
                    qstar.y = 1 / sw * swq.y
                
                    # miu_r
                    s1 = s2 = 0
                    for k in range(self.nPoint):
                        if i == self.oldDotL[k].x and j == self.oldDotL[k].y:
                            continue
                        Pi.x = self.oldDotL[k].x - pstar.x
                        Pi.y = self.oldDotL[k].y - pstar.y
                        PiJ.x = -Pi.y
                        PiJ.y = Pi.x
                        Qi.x = self.newDotL[k].x - qstar.x
                        Qi.y = self.newDotL[k].y - qstar.y
                        s1 += w[k] * (Qi.x * Pi.x + Qi.y * Pi.y)
                        s2 += w[k] * (Qi.x * PiJ.x + Qi.y * PiJ.y)
                    miu_r = np.sqrt(s1**2 + s2**2)
                    # end miu_r

                    curV.x = curV.x - pstar.x
                    curV.y = curV.y - pstar.y
                    curVJ.x = -curV.y
                    curVJ.y = curV.x
                    for k in range(self.nPoint):
                        if i == self.oldDotL[k].x and j == self.oldDotL[k].y:
                            continue
                        Pi.x = self.oldDotL[k].x - pstar.x
                        Pi.y = self.oldDotL[k].y - pstar.y
                        PiJ.x = -Pi.y
                        PiJ.y = Pi.x

                        tmpP.x = ( (Pi.x * curV.x + Pi.y * curV.y) * self.newDotL[k].x -
                                (PiJ.x * curV.x + PiJ.y * curV.y) * self.newDotL[k].y )
                        tmpP.y = ( (-Pi.x * curVJ.x + -Pi.y * curVJ.y) * self.newDotL[k].x +
                                (PiJ.x * curVJ.x + PiJ.y * curVj.y) * self.newDotL[k].y )                     

                        tmpP.x = tmpP.x * w[k] / miu_r
                        tmpP.y = tmpP.y * w[k] / miu_r

                    newP.x = newP.x + tmpP.x    
                    newP.y = newP.y + tmpP.y    

                else:
                    newP = self.newDotL[k]

                if self.preScale:
                    self.rDx[j, i] = newP.x * ratio - i
                    self.rDy[j, i] = newP.y * ratio - j
                else:
                    self.rDx[j, i] = newP.x - i
                    self.rDy[j, i] = newP.y - j

                # end while-j loop
                j += self.gridSize

            # end the while-i loop
            i += self.gridSize
        
        if self.preScale:
            for i in range(self.nPoint):
                self.newDotL[i].x *= 1 / ratio
                self.newDotL[i].y *= 1 / ratio




    def genNewImg(self, oriImg, transRatio):
        # check image channel
        if len(oriImg.shape) == 2:
            channel = 1
            newImg = np.zeros([self.tarH, self.tarW], dtype=np.uint8)
        else:
            channel = oriImg.shape[2]
            newImg = np.zeros([self.tarH, self.tarW, channel], dtype=np.uint8)
        # generate Image base on rDx, rDy
        for i in range(0, self.tarH, self.gridSize):
            for j in range(0, self.tarW, self.gridSize):
                ni = i + self.gridSize
                nj = j + self.gridSize
                w = h = self.gridSize
                # border check
                if ni >= self.tarH:
                    ni = self.tarH - 1
                    h = ni - i + 1
                if nj >= self.tarW:
                    nj = self.tarW - 1
                    w = nj - j + 1
                # bilinear h x w block
                for di in range(h):
                    for dj in range(w):
                        deltaX = bilinear_interp(di/h, dj/w, 
                            self.rDx[i, j], self.rDx[i, nj],
                            self.rDx[ni, j], self.rDx[ni, nj])
                        deltaY = bilinear_interp(di/h, dj/w, 
                            self.rDy[i, j], self.rDy[i, nj],
                            self.rDy[ni, j], self.rDy[ni, nj])
                        nx = j + dj + deltaX * transRatio
                        ny = i + di + deltaY * transRatio
                        # border check
                        nx = max(0, min(nx, self.srcW - 1))
                        ny = max(0, min(ny, self.srcH - 1))
                        # biliear new image
                        nxi = int(nx)
                        nyi = int(ny)
                        nxil = int(np.ceil(nx))
                        nyil = int(np.ceil(ny))

                        if channel == 1:
                            newImg[i+di, j+dj] = bilinear_interp(
                                ny - nyi, nx - nxi, 
                                oriImg[nyi, nxi], oriImg[nyi, nxil],
                                oriImg[nyil, nxi], oriImg[nyil, nxil],
                            )
                        else:
                            for c in range(channel):
                                newImg[i+di, j+dj, c] = bilinear_interp(
                                    ny - nyi, nx - nxi,
                                    oriImg[nyi, nxi, c], oriImg[nyi, nxil, c],
                                    oriImg[nyil, nxi, c], oriImg[nyil, nxil, c],
                                )

        return newImg


    def setAllAndGenerate(self, oriImg, qsrc, qdst, outW, outH, transRatio):
        self.setSize(oriImg.shape[1], oriImg.shape[0])
        self.setTargetSize(outW, outH)
        self.setSrcPoints(qsrc)
        self.setDstPoints(qdst)
        self.calcDelta()
        return self.genNewImg(oriImg, transRatio)
    



if __name__ == "__main__":
    import cv2
    img = cv2.imread('../imgs/girl.jpg')
    srcH, srcW = img.shape[:2]
    print("Height, width ", srcH, srcW)
    outW = srcW
    outH = srcH
    mls = MLS_Rigid()
    import pickle
    with open('points.pkl', 'rb') as f:
        src_point, dst_point = pickle.load(f)
    qsrc = [Point(*sp) for sp in src_point]
    qdst = [Point(*dp) for dp in dst_point]
    new_img = mls.setAllAndGenerate(img, qsrc, qdst, outW, outH, 1)
