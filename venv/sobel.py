
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from cv2 import *

def grayscale(image):
    """
    Transforma una imagen de color a blanco y negro

    :param image:
    :return: image
    """
    input_mage = impread(image) #Convertimos la imagen a su forma matricial
    [nx,ny,nz] = np.shape(image)
    #Extrayendo los componentes de rgb

    ri, gi, bi = input_mage[:,:,0], input_image[:,:,1], input_mage[:,:,2]

    gamma = 1.400 # un parámetro
    rc, gc, bc, = 0.2126, 0.71552, 0.0722 #Pesos para los componentes RGB

    grayscaleImage = rc * ri * gamma + gc*gi ** gamma + bc *bi ** gamma

    fig1 = plt.figure(1)
    ax1, ax2 = fig1.add_subplot(121), fig1.add_subplot(122)
    ax1.imshow(image)
    ax2.imshow(grayscaleImage)
    ax2.imshow(grayscaleImage, cmap=plt.get_cmap('gray'))
    fig1.show()

    return grayscaleImage


def sobel(inputIm):

    #Definimos el kernel de sobel
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


    '''np.shape(grayscaleImage)
    rows = np.shape(grayscaleImage)[0]
    columns = np.shape(grayscaleImage)[1]'''
    print(np.shape(inputIm))
    [rows, columns] = np.shape(inputIm)

    sobelFiltered = np.zeros(shape=(rows, columns)) #Creamos una matriz vacía que serpa nuestra imagen con el filtro
    # Aplicamos el kernel en direcciones x , y
    for i in range(columns - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, inputIm[i:i + 3, j:j +3]))
            gy = np.sum(np.multiply(Gy, inputIm[i:i + 3, j:j +3]))
            sobelFiltered[i + 1, j +1] = np.sqrt(gx ** 2 + gy ** 2) #Calcular "hipotenusa"

    # Mostramos la imagen original y la imagen con el filtro sobel
    fig2 = plt.figure(2)
    ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
    ax1.imshow(inputIm)
    ax2.imshow(sobelFiltered, cmap=plt.get_cmap('gray'))
    fig2.show()


    return sobelFiltered


def reduceImage(image):
    dimensions = np.shape(image)
    print("Dimensiones actuales de la foto: ", image.size, "\nSe reducirá a un cuarto de su tamaño ")
    print(type(image.size))
    photoCrop = photo.resize((640, 480), Image.LANCZOS) #La escalamos a un cuarto de su valor original
    photoCrop.save("/home/guerrero/Downloads/PhotoReduced.png")
    print("Nuevas dimensiones de la foto: ", photoCrop.size)
    return photoCrop

def main():

   ''' # Capturamos una foto
    cam = VideoCapture(1)  # 1 : Especifico el número de cámara (se enceuntra en /dev)
    photo = cam.read()
    # Tomamos la foto'''
   #os.system("sudo fswebcam -r 1280x960  photo.jpg video1 --device video1") #Uso os para no utilizar CV
    
photo = Image.open("/home/guerrero/Downloads/arrow.png")
#photo = Image.open("/home/guerrero/Downloads/street.jpg")

photo = reduceImage(photo)

borderPhoto = sobel(np.asarray(photo))



if __name__ == "__main__":
    main()
