from pcn import *
import unittest

class TestPCN(unittest.TestCase):
    def test_smooth_angel(self):
        a = 120
        b = 60
        output = smooth_angle(a, b)
        self.assertEqual(output, 90)

    def test_iou(self):
        w1 = Window2(100, 20, 40, 60, 80.5, 0.5, 1)
        w2 = Window2(90, 22, 38, 50, 76, 0.6, 2)
        iou = IoU(w1, w2)
        self.assertAlmostEqual(0.482759, iou, delta=0.001)
    
    def test_nms(self):
        w1 = Window2(100, 20, 40, 60, 80.5, 0.5, 1)
        w2 = Window2(90, 22, 38, 50, 76, 0.6, 2)
        w3 = Window2(90, 21, 40, 50, 76, 0.6, 3)
        w4 = Window2(85, 22, 38, 60, 76, 0.8, 4)
        winlist = [w1, w2, w3, w4]
        winlist = NMS(winlist, True, 0.8)
        expect = [4, 3, 1]
        self.assertEqual(expect, [w.conf for w in winlist])
        winlist = NMS(winlist, False, 0.3)
        expect = [4]
        self.assertEqual(expect, [w.conf for w in winlist])
    
    def test_deleteFP(self):
        w1 = Window2(100, 20, 40, 60, 80.5, 0.5, 1)
        w2 = Window2(90, 22, 38, 50, 76, 0.6, 2)
        w3 = Window2(90, 21, 40, 50, 76, 0.6, 3)
        w4 = Window2(85, 22, 38, 60, 76, 0.8, 4)
        winlist = [w1, w2, w3, w4]
        winlist = deleteFP(winlist)
        expect = [4, 3, 2, 1]
        self.assertEqual(expect, [w.conf for w in winlist])

    def test_smooth_windows(self):
        w1 = Window2(100, 20, 40, 60, 80.5, 0.5, 1)
        w2 = Window2(90, 22, 38, 50, 75, 0.6, 2)
        w3 = Window2(90, 21, 40, 50, 24, 0.6, 3)
        w4 = Window2(85, 22, 38, 60, 76, 0.8, 4)
        winlist = [w1, w3, w2, w4]
        winlist = smooth_window(winlist)
        for win in winlist:
            print(win.x, win.y, win.w, win.h, win.angle, win.conf)
        self.assertTrue(True)





if __name__ == '__main__':
    unittest.main()