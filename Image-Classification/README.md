# Image Captioning with ResNet50 Encoder and RNN Decoder

This project implements an image captioning pipeline using a pre-trained ResNet50 CNN as an encoder and an RNN as a decoder to generate captions for images.

## Overview

The goal of this project was to train a model that can generate relevant and coherent captions for images based on the reference captions.
Workflow:
- Feature extraction using Encoder-CNN (ResNet50)
- Caption pre-processing and vocabulary construction
- Building a RNN-decoder (LSTM)
- Training the model
- Generating captions for random images from the dataset
- Evaluating using BLEU score and Cosine Similarity

## Dataset 

This project uses a subset of the COCO "Common Objects in Context" dataset for image caption generation. COCO contains 330K images of 80 object categories and at least five textual reference captions per image. Our subset consists of nearly 5070 of these images, each with five or more different descriptions of the salient entities and activities.

## Model used

**EncoderCNN (ResNet50)**
- Used **ResNet50** pre-trained on ImageNet.
- Removed the final classification layer so our model only extracts features from the images.
- Then resized images to (128, 128) to reduce computational load while retaining sufficient visual quality.
- Applied standard ImageNet normalization: mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) for consistency with the pre-trained ResNet50 expectations.
- Saved the extracted features to disk for efficient training.

**DecoderRNN (LSTM)**
- Designed a RNN-based decoder (LSTM), to generate captions based on image features and reference captions.
- Used an embedding layer to convert tokenized caption words into dense vector representations of size embed_size = 128. (lowered the embed size and hidden size to make it less computationally heavy)
- Resized the extracted image features using a fully connected Linear layer to match the embed_size, followed by BatchNorm1d to stabilize and accelerate training.
- Concatenated the image feature vector (as the first input step) with the embedded caption tokens.
- Fed the resulting sequence into an LSTM layer (hidden_size = 256, num_layers = 1) which processes the temporal word sequence.
- Output of the LSTM is passed through a final linear layer that maps the hidden state at each time step to a vocabulary-sized logits vector, used to predict the next word.
- Used pack_padded_sequence for efficient batching of variable-length captions, ensuring padded tokens do not affect learning.


## Training
- Trained for 10 epochs
-  Used CrossEntropyLoss and Adam optimizer
-  Tracked training and validation loss and accuracy
The best performance was at epoch 5, where: Training Loss: 0.59 | Validation Loss: 0.74 | Training Accuracy: 87.18% | Validation Accuracy: 85.54%
- Plotted the loss/accuracy curves to visualize performance

## Evaluation
- Generated captions for random images in the dataset.
- Used BLEU Score and Cosine Similarity for quantitative evaluation

## Findings
- Average BLEU Score: 0.5361, suggests moderate similarity with reference captions
- Average Cosine Similarity: 0.3239 (rescaled to 0.66), reflects semantic similarity based on embeddings
- Rescaling both scores showed they aligned better, but further tuning is required for improved generalization.
- BLEU score is limited in capturing semantic meaning—it only checks word overlap.
- Cosine similarity better evaluates meaning, but may penalize paraphrased captions.
- When both scores agree, captions tend to be high quality.
- When they differ significantly, the caption might still be valid but phrased differently.

## Future Improvements
- Fine-tune the decoder with attention mechanisms
- Experiment with Transformer-based decoders
- Train on larger datasets for better generalization
- Perform more hyperparameter tuning (embedding size, hidden size, learning rate)

Also created a function at the end for reusability, for similar image-captioning models, which trains the model and validates it and plots graphs to visualize the loss and accuracy.













