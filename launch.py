import gradio as gr
import numpy as np
import cv2
from tensorflow import keras

model = keras.models.load_model('DinoSnapModel.h5')

labels = list(['Ankylosaurus', 'Brachiosaurus', 'Parasaurolophus',
       'Spinosaurus', 'Stegosaurus', 'T-Rex',
       'Triceratops', 'Velociraptor'])

IMG_WIDTH = 80
IMG_HEIGHT = 80


def classify_dino(inp):

  img_gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)/255.
  img_resized = cv2.resize(img_gray,(IMG_WIDTH,IMG_HEIGHT))
  prediction = model.predict(img_resized.reshape(1,IMG_WIDTH,IMG_HEIGHT,1))
  return {labels[i]: float(prediction[0,i]) for i in range(len(labels))}

image = gr.inputs.Image(shape=(IMG_WIDTH, IMG_HEIGHT))
label = gr.outputs.Label(num_top_classes=3)
sample_images = [['examples/ex1_ankylosaurus.JPG'],['examples/ex1_spinosaurus.JPG'],['examples/ex1_stegasaurus.JPG']
                 ,['examples/ex1_trex.JPG'],['examples/ex1_triceratops.JPG']]
title = 'Dino Detective!'
description = 'Upload a picture of a dinosaur (or choose one from the samples below) and learn which one it is. Currently classifying only 8 types of dinosaurs (Ankylosaurus, Brachiosaurus, Parasaurolophus, Spinosaurus Stegosaurus T-Rex Triceratops Velociraptor). For best results, place the toy on a flat surface and take a picture of it against a flat background.'

gr.Interface(fn=classify_dino, inputs=image, outputs=label, capture_session=True,examples=sample_images, title=title, description=description).launch(share=True)
