import tflite_runtime.interpreter as tflite
import numpy as np


class GestureClassifier:
    def __init__(
        self,
        model_path: str,
        num_threads=1,
    ):
        self.interpreter = tflite.Interpreter(model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        self.interpreter.set_tensor(
            self.input_details[0]["index"], np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()

        result = self.interpreter.get_tensor(self.output_details[0]["index"])
        
        result_squeeze = np.squeeze(result)

        index = np.argmax(result_squeeze)
        confidence = np.max(result_squeeze)

        return index, confidence
