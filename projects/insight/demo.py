import sys
import insight


img = sys.argv[1]
x = insight.estimate(img)
print(x)