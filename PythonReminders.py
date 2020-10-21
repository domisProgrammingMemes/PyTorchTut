import os

if __name__ == "__main__":
    # for files (e.g. data)
    def pathing():
        entires = os.listdir(".")
        for entry in entires:
            print(entry)
        print(os.listdir("D:/Hochschule RT/Masterthesis/Coding/PyTorchTut"))
    # test prints:
    # path = "." is equal to "root" <- root is where the .py is located at
    print(os.listdir("."))
    # pathing()