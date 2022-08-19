from PIL import Image


image_file = Image.open("/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/real_images/real_our_img (17).jpeg") # open colour image
image_file = image_file.convert('1') # convert image to black and white
image_file.save('/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/outputs/bin/result.png')
