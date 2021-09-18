import gradio as gr
import numpy as np
import cv2
from tensorflow import keras

model = keras.models.load_model('DinoSnapModel.h5')

labels = list(['Ankylosaurus', 'Brachiosaurus', 'Paceacephalasaurus',
       'Parasaurolophus', 'Pterodactyl', 'Spinosaurus',
       'Stegosaurus', 'T-Rex', 'Triceratops',
       'Velociraptor', 'No Dino Found (please try again)'])

IMG_SIZE = 224

def classify_dino(inp):

  img_resize = inp.resize((IMG_SIZE,IMG_SIZE))
  img_np = np.array(img_resize).astype(np.float32)/255.
  np.save('img_np_gradio',img_np)
  prediction = model.predict(img_np.reshape(1,IMG_SIZE,IMG_SIZE,3))
  return {labels[i]: float(prediction[0,i]) for i in range(len(labels))}

image = gr.inputs.Image(type='pil',image_mode="RGB")
label = gr.outputs.Label(num_top_classes=3)
sample_images = [['ex1_ankylosaurus.JPG'],['ex1_spinosaurus.JPG'],['ex1_stegasaurus.JPG']
                 ,['ex1_trex.JPG'],['ex1_triceratops.JPG']]
title = 'Dino Detective!'
<<<<<<< HEAD
description = 'Upload a picture of a dinosaur (or choose one from the samples below) and learn which one it is. \n Currently classifying only (Ankylosaurus, Brachiosaurus, Paceacephalasaurus, Parasaurolophus, Pterodactyl, Spinosaurus, Stegosaurus, T-Rex, Triceratops, Velociraptor)'
=======
description = 'Snap a picture of a dinosaur (or choose one from the samples below) and find out which one it is. Currently classifying only 8 types of dinosaurs (Ankylosaurus, Brachiosaurus, Parasaurolophus, Spinosaurus Stegosaurus T-Rex Triceratops Velociraptor). For best results, place the toy on a flat surface and take a picture of it against a flat background. Feedback? Questions? Email me to dinodetect@gmail.com'
>>>>>>> 8e5590eb07d22a072386e868a1394a0ee9dc3bb0

gr.Interface(fn=classify_dino, inputs=image, outputs=label, capture_session=True,examples=sample_images, title=title, description=description).launch(debug=True)
