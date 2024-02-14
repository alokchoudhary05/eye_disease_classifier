import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Get predictions
        predictions = model.predict(test_image)[0]
        
        # Define categories
        categories = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eye', 'Glaucoma', 'Uveitis']

        # Determine prediction
        max_index = np.argmax(predictions)
        prediction = categories[max_index]

        return [{"image": prediction}]