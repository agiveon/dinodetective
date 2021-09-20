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
sample_images = [['examples/ex1_ankylosaurus.JPG'],['examples/ex1_spinosaurus.JPG'],['examples/ex1_stegasaurus.JPG']
                 ,['examples/ex1_trex.JPG'],['examples/ex1_triceratops.JPG']]
title = 'Dino Detective!'

description = 'Snap a picture of a dinosaur (or choose one from the samples below) and find out which one it is. Currently classifying 10 types of dinosaurs (Ankylosaurus, Brachiosaurus, Paceacephalasaurus, Parasaurolophus, Pterodactyl, Spinosaurus, Stegosaurus, T-Rex, Triceratops, Velociraptor). For best results, place the toy on a flat surface and take a picture of it against a uniform background. Feedback? Questions? Email me to dinodetect@gmail.com'

gr.Interface(fn=classify_dino, inputs=image, outputs=label, capture_session=True,examples=sample_images, title=title, description=description).launch(debug=False)
