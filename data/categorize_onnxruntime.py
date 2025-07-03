import onnxruntime as ort
import numpy as np
from scipy.io import wavfile

# 1) Read audio (e.g. 10 s clip at 16 kHz)
sr, audio = wavfile.read("mix.wav")
audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
# 2) Prepare model
sess = ort.InferenceSession("genre_discogs400-discogs-maest-10s-dw-1.onnx")
# 3) Run (assuming model input name is 'input')
outputs = sess.run(None, {"input": audio.reshape(1, -1)})
# 4) Get top-1 genre from the JSON labels file
labels = [line.strip() for line in open("genre_discogs400-discogs-maest-10s-dw-1.json")]
top1 = labels[int(np.argmax(outputs[0]))]
print("Predicted genre:", top1)
