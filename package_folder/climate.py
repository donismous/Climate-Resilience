import os
import pickle

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

def prediction_function(a, b, c, d):
    """Function using a pretrained model loaded from disk to output a prediction.

    DUMMY ARGUMENTS:
    - a
    - b
    - c
    - d
    """
    # Load the model from the pickle file
    model_path = os.path.join(ROOT_PATH, 'models', 'best_model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Use the model to predict the given inputs
    # prediction = model.predict([[a, b, c, d]])

    #DUMMY
    prediction = a+b+c+d

    return prediction
