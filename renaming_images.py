import os


# Function to rename multiple files
def main(xyz):
    for count, filename in enumerate(os.listdir(xyz)):
        dst = "pituitary_" + str(count) + ".jpg"
        src = xyz + filename
        dst = xyz + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
main("/home/yashas/Desktop/brain_cnn/pituitary/")