import gradio as gr
from fastai.vision.all import *

learn = load_learner("export.pkl")

# Update these to whatever your categories were for training
categories = learn.dls.vocab

def classify_image(img):
    """Takes in an imagem and classifies it using a model"""
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image()
label = gr.Label()

demo = gr.Interface(fn=classify_image, inputs=image, outputs=label)
demo.launch(inline=False)