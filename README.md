---
license: cc-by-nc-4.0
inference: true
tags:
- mms
- vits
pipeline_tag: text-to-speech
language:
- ky
---

# Introduction

This repository contains a text-to-speech (TTS) model fine-tuned on data consisting of sentences in the Kyrgyz language with audio examples voiced by a single speaker. The audio is provided at a sample rate of 16 kHz. The dataset comprises 3500 examples and 4 hours of audio. The model is based on the facebook/mms-tts-kir model pre-trained on the Kyrgyz language. The code for fine-tuning the model was based on the code from this [GitHub repository](https://github.com/ylacombe/finetune-hf-vits). Experimental findings concluded that the best results are achieved through two-stage fine-tuning:

* Training with Learning Rate 1e-4 and 4 epochs,
* Training with Learning Rate 9e-7 and 80 epochs.


# MMS: Scaling Speech Technology to 1000+ languages

The Massively Multilingual Speech (MMS) project expands speech technology from about 100 languages to over 1,000 by building a single multilingual speech recognition model supporting over 1,100 languages (more than 10 times as many as before), language identification models able to identify over [4,000 languages](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html) (40 times more than before), pretrained models supporting over 1,400 languages, and text-to-speech models for over 1,100 languages. Our goal is to make it easier for people to access information and to use devices in their preferred language.  

You can find details in the paper [Scaling Speech Technology to 1000+ languages](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/) and the [blog post](https://ai.facebook.com/blog/multilingual-model-speech-recognition/).

An overview of the languages covered by MMS can be found [here](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html).

##  Transformers

MMS has been added to Transformers. For more information, please refer to [Transformers' MMS docs](https://huggingface.co/docs/transformers/main/en/model_doc/mms).

[Click here](https://huggingface.co/models?other=mms) to find all MMS checkpoints on the Hub. 

Checkout the demo here [![Open In HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/facebook/MMS) 

## 


# Inference


The model takes Cyrillic text in the Kyrgyz language as input and preprocesses it by removing punctuation marks (periods, commas, colons, exclamation and question marks) as well as words written in Latin script. Therefore, it is not advisable to feed multiple sentences into the model at once as they will be vocalized without intonational pauses, indicating the end of one and the beginning of a new sentence. Words written in Latin script will be skipped in the generated speech.

For example:
```
text = 'Кандай улут болбосун кыргызча жооп кайтарышыбыз керек.'
```

You can use this model by executing the code provided below.

```
import subprocess
from transformers import pipeline
from IPython.display import Audio
import numpy as np
import torch
import scipy

model_id = "Simonlob/simonlob_akylay"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

```

```
text = 'Кандай улут болбосун кыргызча жооп кайтарышыбыз керек.'
speech = synthesiser(text)
```
The output of the model looks as follows:
```
{'audio': array([[-1.7045566e-04,  8.9107212e-05,  2.8329418e-04, ...,
          8.0898666e-08,  4.8763245e-06,  5.4663483e-06]], dtype=float32),
 'sampling_rate': 16000}
```

Listen to the result:

```
Audio(speech['audio'], rate=speech['sampling_rate'])
```

Save the audio as a file:

```
scipy.io.wavfile.write("<OUTPUT PATH>.wav", rate=speech["sampling_rate"], data=speech["audio"][0])
```



</details>

## Model details

- **Model type:** Text-to-speech model
- **License:** CC-BY-NC 4.0 license
- **Cite as:**

      @article{pratap2023mms,
        title={Scaling Speech Technology to 1,000+ Languages},
        author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
      journal={arXiv},
      year={2023}
      }


## Credits


- Facebook AI Research ([Official Space](https://huggingface.co/spaces/facebook/MMS))
- Yoach Lacombe (Research) [GitHub](https://github.com/ylacombe/finetune-hf-vits)
- The Cramer Project (Data collection and preprocessing)[Official Space](https://thecramer.com/), [Akyl_AI](https://github.com/Akyl-AI)
- Amantur Amatov (Expert)
- Timur Turatali (Expert, Research)
- Den Pavlov (Research, Data preprocessing and fine-tuning) [GitHub](https://github.com/simonlobgromov/finetune-hf-vits)
- Ulan Abdurazakov (Environment Developer)
- Nursultan Bakashov (CEO)


(Training and fine-tuning)
- Acme Corp (Financial support and resources)


## Additional Links

- [Blog post](https://ai.facebook.com/blog/multilingual-model-speech-recognition/)
- [Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/mms).
- [Paper](https://arxiv.org/abs/2305.13516)
- [GitHub Repository for fine tuning](https://github.com/ylacombe/finetune-hf-vits)
- [GitHub Repository](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr)
- [Other **MMS** checkpoints](https://huggingface.co/models?other=mms)
- MMS base checkpoints:
  - [facebook/mms-1b](https://huggingface.co/facebook/mms-1b)
  - [facebook/mms-300m](https://huggingface.co/facebook/mms-300m)
