
from os import listdir
from os.path import isfile, join
import os, random
import cv2
import numpy as np




train_path = "data/high_res/train"
valid_path = "data/high_res/valid"
test_path= "data/high_res/test"
fake_path = "data/high_res/manip"

output_file_path = "data/test.txt"

train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
test_files  = [f for f in listdir(test_path) if isfile(join(test_path,  f))]
valid_files = [f for f in listdir(valid_path) if isfile(join(valid_path, f))]

manip_files = [f for f in listdir(fake_path) if isfile(join(fake_path, f))]


def make_image(main_image, manip_image,loc):
    
    # Load images
    base_image = cv2.imread(os.path.join(loc,main_image))
    airplane_image = cv2.imread(os.path.join(fake_path,manip_image), -1)  # -1 to load with alpha channel

    # Define position
    x = random.randint(700, 1300)  # Horizontal position
    y = random.randint(1000, 1600)   # Vertical position

    # Extract the dimensions of the airplane image
    try:
        rows, cols, channels = airplane_image.shape[:3]
    except:
        print(os.path.join(fake_path,manip_image))
        

    # Ensure the airplane fits within the base image dimensions
    if y + rows > base_image.shape[0] or x + cols > base_image.shape[1]:
        raise ValueError("Airplane image goes out of base image bounds.")

    # Extract the region of interest (ROI) from the base image
    roi = base_image[y:y+rows, x:x+cols]

    # Extract the alpha mask and its inverse from the airplane image
    airplane_mask = airplane_image[:, :, 3]
    airplane_mask_inv = cv2.bitwise_not(airplane_mask)

    # Create the foreground and background images
    airplane_fg = cv2.bitwise_and(airplane_image[:, :, :3], airplane_image[:, :, :3], mask=airplane_mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=airplane_mask_inv)

    # Overlay the airplane onto the base image
    dst = cv2.add(img1_bg, airplane_fg)
    base_image[y:y+rows, x:x+cols] = dst
    
    # Save or display the result
    cv2.imwrite(os.path.join(loc+"_manip",main_image), base_image) #############

    # Save the binary mask
    # Resize the mask to the size of the base image
    full_mask = np.zeros_like(base_image[:, :, 0])
    full_mask[y:y+rows, x:x+cols] = airplane_mask
    
    parts = main_image.split("_")

    # The first part contains "abc"
    mask_ = parts[0]
    mask_name = mask_+ "_mask.jpg"
    cv2.imwrite(os.path.join(loc+"_manip",mask_name), full_mask)  ##############
    
    with open(output_file_path, 'a') as file:
        file.write(f"{os.path.join(loc+'_manip', main_image)},{os.path.join(loc+'_manip', mask_name)}\n")
    
    return 


for f in test_files:
    if "mask" not in f:
        manip_image = random.choice(manip_files)
        make_image(f, manip_image,test_path)

