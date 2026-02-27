import tensorflow as tf
from tensorflow.keras.layers import InputLayer as KerasInputLayer
from tensorflow.keras import mixed_precision


class LegacyInputLayer(KerasInputLayer):
    """InputLayer that is compatible with older Keras configs.

    Older .h5 models sometimes store `batch_shape` instead of
    `batch_input_shape`. This class maps that key so that models
    saved with old Keras can be deserialized on newer versions.
    """

    @classmethod
    def from_config(cls, config):
        # Map legacy key if present
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = tuple(config.pop("batch_shape"))
        return super().from_config(config)


custom_objects = {"InputLayer": LegacyInputLayer}

# Map legacy DTypePolicy objects used in older Keras versions directly
# to the current mixed_precision.Policy class.
custom_objects["DTypePolicy"] = mixed_precision.Policy

model = tf.keras.models.load_model(
    "fabric_defect_model.h5", custom_objects=custom_objects, compile=False
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # for quantization, optional

tflite_model = converter.convert()

with open("fabric_defect_model_quant_new.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved fabric_defect_model_quant_new.tflite")