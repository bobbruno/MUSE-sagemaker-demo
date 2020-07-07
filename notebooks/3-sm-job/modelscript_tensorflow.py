import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import json

#Return loaded model
def load_model(modelpath):
    model = hub.load(modelpath)
    return model

# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    if not isinstance(payload, str):
        payload = payload.decode()
    try:
        try:
            if isinstance(json.loads(payload), dict):
                data = json.loads(payload).get('instances', [payload])  # If it has no instances field, assume the payload is a string
            elif isinstance(json.loads(payload), list):
                data = json.loads(payload)
            else:
                raise json.JSONDecodeError
        except json.JSONDecodeError:  # If it can't be decoded, assume it's a string
            data = [payload]
        result = model(data)['outputs'].numpy()
        out = result.tolist()
    except:
        the_type, the_value, _ = sys.exc_info()
        out = f"{the_type}: {the_value}: {str(payload)}"
    return json.dumps({'output': out})