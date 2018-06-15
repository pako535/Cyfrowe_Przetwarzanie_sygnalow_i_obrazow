import cv2
import numpy as np
from matplotlib import pyplot as plt
import imghdr

# wczytanie obrazu
path = "../standard_test_images/jetplane.tif"
img = cv2.imread(path)

cv2.imshow("tytul okna", img)  # stworzenie okna z obrazkiem

cv2.waitKey()  # ważna funkcja!!!
cv2.destroyAllWindows()  # zamknięcie wszystkich okien
# Funkcja waitKey() odpowiada za obsługę okienek (ich rysowanie/odświeżanie/interakcję z użytkownikiem).
#  Funkcja zwraca kod klawisza który został naciśnięty.


height, width, channels = img.shape  # zczytanie parametrów wysokosc, szerokosc ilosc kanałów
print("Wysokość: %s,\nSzerokość: %s,\nIlosc pikseli: %s,\nIlosc kanałów: %s,\nTyp danych obrazu: %s,\nTyp pliku: %s"
      % (height, width, img.size, channels, img.dtype, imghdr.what(path)))

# stworzenie maski
mask = np.zeros(img.shape[:2], np.uint8)
mask[150:350, 100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)

# Wyliczenie histogramu bez użytej maski
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# wyświetlenie paru obrazków na jednym "plot'cie"
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_full)
plt.xlim([0, 256])

plt.show()
