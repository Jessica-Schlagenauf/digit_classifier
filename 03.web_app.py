from gradio.components import Image
from gradio.components import Label
from gradio import Interface

from sklearn.preprocessing import MinMaxScaler 

import joblib 

model=joblib.load(filename='models/digit_model.joblib')

input_image = Image(shape=(8,8), image_model='L', invert_colors=True, source='canvas', label='INPUT DIGIT')
output_labels = Label(num_top_classes=10, label='MODEL PREDICTION' )
title='Digit classifier with ML'
description='This project is a demo for the class fo TAG DS&AI master. '


def predict_image(image):
    flat_image = image.reshape(-1,64)
    print(flat_image)
    return None


interface= Interface(
    fn=predict_image, 
    inputs=input_image, 
    outputs=output_labels, 
    title=title, 
    description=description,
    )
interface.launch()



