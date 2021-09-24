import gradio as gr
import torch
from torchvision import transforms
import requests
from PIL import Image

model = torch.load('DinoTransModel.pt')
model.eval()

labels = ['Ankylosaurus', 'Brachiosaurus', 'Paceacephalasaurus', 'Parasaurolophus', 'Pterodactyl', 'Spinosaurus', 'Stegosaurus', 'T-Rex', 'Triceratops', 'Velociraptor']

def classify_dino(img):

  img = transforms.Resize((224,224))(img)
  img = transforms.ToTensor()(img)
  img = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img)

  with torch.no_grad():
    new_pred = model(img.view(1,3,224,224)).argmax()
  return labels[new_pred.item()]

image = gr.inputs.Image(type='pil',image_mode="RGB")
label = gr.outputs.Label()
title = 'Dino Detective!'
description = 'Upload a picture of a dinosaur (or choose one from the samples below) and learn which one it is. \n Currently classifying only (Ankylosaurus, Brachiosaurus, Paceacephalasaurus, Parasaurolophus, Pterodactyl, Spinosaurus, Stegosaurus, T-Rex, Triceratops, Velociraptor)'

gr.Interface(fn=classify_dino, inputs=image, outputs=label, capture_session=True, title=title, description=description).launch(debug=True)
