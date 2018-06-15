import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "../standard_test_images/jetplane.tif"
img = cv2.imread(path)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()

# maskowanie tablicy do danej wartosci
# Możesz zobaczyć histogram leży w jaśniejszym regionie.
# Potrzebujemy pełnego spektrum. Do tego potrzebujemy funkcji transformacji,
# która mapuje piksele wejściowe w jaśniejszym regionie, aby wyprowadzać piksele w pełnym obszarze
cdf_m = np.ma.masked_equal(cdf, 0)
# Teraz znajdujemy minimalną wartość histogramu (z wyłączeniem 0) i stosujemy równanie wyrównania histogramu
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
cdf_normalized = cdf * hist.max() / cdf[-1]

img = cdf[img]
plt.hist(img.flatten(), 256, [0, 256], color="r")
plt.xlim([0, 256])
plt.show()



