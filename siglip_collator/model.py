# MIT License
#
# Copyright (c) 2024 Tsutomu Furuse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from transformers import AutoProcessor, AutoModel
import torch


HF_MODEL = "google/siglip-base-patch16-256-multilingual"


class SiglipModel():

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() \
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(HF_MODEL)
        self.processor = AutoProcessor.from_pretrained(HF_MODEL)

    def infer(self, image, texts):
        image = image.convert("RGB")
        inputs = self.processor(
            text=texts, images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        return logits_per_image, probs