import numpy as np
import argparse
import imutils
import cv2

# zdeklarowanie wymaganych argumentów dla danego skryptu
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-o", "--output", required=True,
                help="path to the output image")
args = vars(ap.parse_args())

# licznik znalezionych elementów
counter = 0

# wczytanie zdjęcia do programu (sciezka podana w argumencie skryptu)
image_orig = cv2.imread(args["image"])
height_orig, width_orig = image_orig.shape[:2]

# wyjsciowe zdjecie z konturami
image_contours = image_orig.copy()

# zdeklarowanie ilosci kolorów
colors = ['1', '2']
for color in colors:

    # zrobienie kopi oryginalnego zdjęcia
    image_to_process = image_orig.copy()
    image_to_process2 = image_orig.copy()

    # zdefiniowanie tablic NumPy granic kolorów (wektory GBR)
    if color == '1':
        lower = np.array([60, 100, 20])
        upper = np.array([240, 240, 210])

    elif color == '2':
        # odwrócenie kolorów obrazu
        image_to_process = (255 - image_to_process)
        lower = np.array([50, 50, 40])
        upper = np.array([100, 120, 80])

    # poszukiwanie kolorów w okreslonych granicach
    image_mask = cv2.inRange(image_to_process, lower, upper)
    # zastosowanie maski
    image_res = cv2.bitwise_and(image_to_process, image_to_process, mask=image_mask)

    # wczytanie obrazu, przekonwertowanie go na skalę szarości i rozmycie
    image_gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # wykonanei wykrywania krawędzi, wykonanie dylatacji + erozja, aby zamknąć szczeliny między krawędziami obiektu
    image_edged = cv2.Canny(image_gray, 50, 100)
    image_edged = cv2.dilate(image_edged, None, iterations=1)
    image_edged = cv2.erode(image_edged, None, iterations=1)

    # znalezienie konturów w mapie krawędzi
    cnts = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
        # jeśli kontur nie jest wystarczająco duży, zignoruj go (<5)
        if cv2.contourArea(c) < 5:
            continue

        # wyliczenie otoczki wypukłej skończonego zbioru punktów płaszczyzny dla kontur
        # funkcja korzysta algorytmu Grahama dla otoczki wypukłem
        hull = cv2.convexHull(c)

        # wyrysowanie konturów na kopi obrazu
        cv2.drawContours(image_contours, [hull], 0, (0, 255, 0), 1)
        counter += 1

# wypisanie ilosci obiektó
print("{} obiektów".format(counter))
# zapisannie obrazu z konturami
cv2.imwrite(args["output"], image_contours)
