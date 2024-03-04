import subprocess
from transformers import pipeline
from IPython.display import Audio
import numpy as np
import torch
import scipy

model_id = "Simonlob/simonlob_akylay"
synthesiser = pipeline("text-to-speech", model_id)
