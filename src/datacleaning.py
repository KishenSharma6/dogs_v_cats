import os, shutil, random

def move_images(path_to_read, path_to_write, images):
    """
    move subsample of images from train -> exp 
    """
    files = os.listdir(path_to_read)

    for i, file in enumerate(files):
        if file in images:
            shutil.copy2(path_to_read + file, path_to_write)
    print("files moved successfully")

def rename_to_idx(path):
    files = os.listdir(path)
    for i, file in enumerate(files):
        os.rename(path + file, path + str(i) + '.jpg')
    print("rename completed")

def select_images(n, label):
    #generate 50 numbers less than 12_500 - cats
    if label == 0:
        cats  = [str(random.randint(0, 12_500)) + '.jpg' for i in range(0, n)]
        return cats

    #generate numbers more than 12_500 - dogs
    elif label == 1:
        dogs  = [str(random.randint(12_500, 25_000)) + '.jpg' for i in range(0, n)]
        return dogs
    
    else:
        raise Exception("incorrect label!")

def standardize_image_name(path, cat_start_idx= 0, dog_start_idx= 12_500):
    """
    convert image names to numeric names
    """
    cat_idx = cat_start_idx
    dog_idx = dog_start_idx
    for img in os.listdir(path):
        if img.startswith('cat.'):    
            os.rename(os.path.join(path, img), os.path.join(path, str(cat_idx) + ".jpg"))
            cat_idx +=1
        if img.startswith('dog.'):    
            os.rename(os.path.join(path, img), os.path.join(path, str(dog_idx) + ".jpg"))
            dog_idx +=1