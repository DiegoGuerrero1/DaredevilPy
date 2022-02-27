
def main():


    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    img = Image.open("/home/guerrero/Documents/UNAM/4th-SEMESTER/COMP-II/Parcial1/daredevilPy/cube.png")
    print(type(img))
    img.show()


    num_img = np.asarray(img)
    numi:size = np.array([len(num_img), len(num_img[0])])
    print(type(num_img))
    print(num_img)





if __name__ == "__main__":
    main()
