import os
import re

from PIL import Image

dirr="/home/student/star_tracker/star_images_new/"
save_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/star-images/"
directory = os.fsencode(dirr)
num_v=5001    
for f in os.listdir(directory):
    filename = os.fsdecode(f)

    if filename.endswith(".png") : 
      img = Image.open(dirr+str(filename)) # Load the image

      u=filename.split("_")
      v=filename.split(".")
      print(u,v)


      f_new=u[0]+"-"+str(num_v)+".png"
      img.save(save_dir+f_new) # Save the image
      num_v+=1
