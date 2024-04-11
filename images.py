import os
import requests
from PIL import ImageOps 
import numpy as np
from PIL import Image 

import matplotlib.pyplot as plt

# List of image URLs to download
urls = [
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png',
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png',
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png'
]

# Download images from URLs
for url in urls:
    response = requests.get(url, stream=True, timeout=5)
    filename = url.rsplit('/', maxsplit=1)[-1]
    with open(filename, 'wb') as out_file:
        out_file.write(response.content)
    del response

def get_concat_h(im1, im2):
    """
    Concatenates two images horizontally.
    """
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

MY_IMAGE = "lenna.png"

# The "path" to an image can be obtained using the current working directory
cwd = os.getcwd()
print("working directory is : ",cwd) 

# Load images
image_path = os.path.join(cwd, MY_IMAGE)
print("path to image is ", image_path)

# Load the image using the image's filename and create a PIL Image object
image = Image.open(image_path)
type(image)
print("path to image is " ,image_path)

# Plotting the image
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.title("Lena : plotting image")
plt.show()

print("image size is  ",image.size)
print("image mode is  ",image.mode)

# Load image into memory 
im = image.load() 
# Check intensity at specific location
x = 0
y = 1
print('intensity is ',im[y,x])

# Save image as JPEG
image.save("lenna.jpg")

# Convert image to grayscale
image_gray = ImageOps.grayscale(image) 
image_gray 
print('image mode is ',image_gray.mode)

# Quantization
image_gray.quantize(256 // 2)
print('image quantization levels are ',image_gray.quantize(256 // 2).getcolors())
image_gray.show()

get_concat_h(image_gray,  image_gray.quantize(256//2)).show(title="Lena") 
for n in range(3,8):
    plt.figure(figsize=(10,10))
    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) 
    plt.title(f"256 Quantization Levels  left vs {256//2**n}  Quantization Levels right")
    plt.show()

# Color Channels
baboon = Image.open('baboon.png')
print('baboon image mode is ',baboon.mode)
baboon

red, green, blue = baboon.split()

get_concat_h(baboon, red)
print('red channel mode is ',red.mode)
get_concat_h(baboon, blue)
print('blue channel mode is ',blue.mode)
get_concat_h(baboon, green)
print('green channel mode is ',green.mode)

# PIL Images into NumPy Arrays
array= np.asarray(image)
print('pill array set numpy: ',type(array))

array = np.array(image)
# Summarize shape
print('image shape', array.shape)
 

array[0, 0]
array.min()
array.max()
print('array min is ',array.min())
print('array max is ',array.max())

# INDEXING
print('indexing array is ',array[0,0])
plt.figure(figsize=(10,10))
plt.imshow(array)
plt.title("Original Image -indexing")
plt.show()

ROWS = 256
COLUMNS = 256

plt.figure(figsize=(10,10))
plt.imshow(array[:,0:COLUMNS,:])
plt.title("Cropped Image")
plt.show()

A = array.copy()
plt.imshow(A)
plt.title("A")
plt.grid()
plt.xlabel("X")
plt.ylabel("Y")
plt.legend("Legend")
plt.show()

B = A
A[:,:,:] = 0
plt.imshow(B)
plt.grid()
plt.xlabel("X")
plt.ylabel("Y")
plt.legend("Legend")
plt.title("B")
plt.show()

baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array)
plt.title("Original Image")
plt.show()

baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array[:,:,0], cmap='gray')
plt.title("Grey Channel")
plt.show()

baboon_red=baboon_array.copy()
baboon_red[:,:,1] = 0
baboon_red[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_red)
plt.title("Red Channel")
plt.show()

baboon_blue=baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.title("Blue Channel")
plt.show()

baboon_green=baboon_array.copy()
baboon_green[:,:,0] = 0
baboon_green[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_green) 
plt.title("Green Channel")
plt.show()

# Cropping Images
image_cropped = image.crop((100, 100, 300, 300))
plt.figure(figsize=(10,10))
plt.imshow(image_cropped)
plt.title("Cropped Image")
plt.show()


#Cropped Image Information
print('cropped image information:')
print("image_cropped.format is ",image_cropped.format)
print("image cropped size is ",image_cropped.size)
print("image cropped mode is ", image_cropped.mode)
print("image cropped box is ", image_cropped.getbbox())
print("image cropped colors are, ", image_cropped.getcolors())
print("image cropped extrema are : ", image_cropped.getextrema())
print("image cropped data is : ",image_cropped.getdata(),)
print("image cropped Bands are : ",image_cropped.getbands())
print("image cropped pixels are: ",image_cropped.getpixel((0, 0)))
print("image cropped palette is : ",image_cropped.getpalette())
print("image cropped historigram is : ", image_cropped.histogram())
print("image cropped info is : ", image_cropped.info)

