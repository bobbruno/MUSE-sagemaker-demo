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
        except json.JSONDecodeError:  # If it can't be decoded, assume it's a string
            data = [payload]
        result = np.asarray(model(data))
        out = result.tolist()
    except Exception as e:
        out = str(e)
    return json.dumps({'output': out})