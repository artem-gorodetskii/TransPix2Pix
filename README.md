# TransPix2Pix
We propose a novel model for sketch-to-photo translation capable of  producing high fidelity realistic images with style adaptation. Inspired by the performance of the [pix2pix](https://arxiv.org/pdf/1611.07004.pdf) conditional GAN and [Vision Transformer](https://arxiv.org/pdf/2010.11929.pdf), we have developed a hybrid architecture, referred to as TransPix2Pix, that learns spatial and latent dependencies between convolutional features in the image-to-image translation tasks. To perform style adaptation the proposed model makes use of a separately pretrained style encoder, which converts image characteristics into an embedding vector. We apply the developed model for converting sketch images of cats into realistic photos and demonstrate that the proposed approach is more effective than the conventional one in terms of image quality.

### Repository structure:
- **model_generator.py** and **model_discriminator.py** contain TransPix2Pix Generator and Disrimiator models, respectively.
- **transformer_block.py** and **attention_block.py** contain TransformerBlock and Self-AttentionBlock.
- **loss_network.py** includes a network that used for feature reconstruction loss computation.
- **model_ema.py** contains a class, that performs exponential mooving average of models weights.
- The **StyleEncoder** directory includes the Style Encoder model and jupyter notebook with training and inference functions.

### Dataset
We train models on a custom dataset based on [Kaggle Cat Dataset](https://www.kaggle.com/datasets/crawford/cat-dataset). To prepare sketch-photo pairs we develop a preprocessing pipeline that includes the following steps:
- Cat segmentation using a pretrained [segmentation model](https://github.com/WillBrennan/SemanticSegmentation).
- Sketch generation using a pretrained [photo-to-sketch model](https://github.com/mtli/PhotoSketch).
- Sketch processing using a developed algorithm (to make similar and thin sketches).
- Image cropping and resizing to 256x256  resolution.

The style encoder is trained on a subset of the prepared dataset separated into 9 different cat styles including: black, black-white, gray, siberian, siamese, ginger, ginger-white, white and “other”. The “other” class contains cat styles which do not belong to any other category.

### Style Encoder
The style encoder represents a convolutional network that encodes cat images into style feature embeddings. The network is based on the [CosFace](https://arxiv.org/pdf/1801.09414.pdf) architecture and composed of a set of convolution blocks followed by the fully connected layers. The convolution part is organized as follows:
- The model consists of 4 convolution blocks.
- Each convolutional layer is followed by BatchNormalization and ReLU.
- We apply residual connections between each couple of adjacent layers with the same dimension.
- The first block consists of 2 convolutional layers with SpatialDropout2D between them.
- Blocks 2, 3 and 4 contain 3 convolutional layers, with the first layer having a stride of 2 to reduce the spatial dimension. SpatialDropout2D is used after the second convolutional layer.
- The number of feature filters has an initial value of 32 and is doubled for each block.
- We apply Dropout to the output of each convolution block.

The output of the last block is flattened and processed by two linear layers (with BatchNormalization) to form an embedding vector. The obtained embedding is passed through the CosFace layer with the SoftMax function. During training, the network is optimized to minimize the cross-entropy loss function. 

The proposed style encoder operates in a 256-dimensional latent space.  We have trained several versions of the model and determined that this configuration leads to better performance in terms of classification accuracy and embedding quality. To evaluate the quality of constructed latent space we estimate equal error rates, which are computed using cosine similarity scores by pairing each embedding vector with each class centroid. Figure 1 shows the [U-Map](https://arxiv.org/pdf/1802.03426.pdf) projection of the computed style embeddings.


<p align="center">
  <img alt="img-name" src="assets/umap_projection.png" width="200">
  <br>
    <em>Fig. 1. U-Map projection of the computed style embeddings. Each color corresponds to a different cat style.</em>
</p>

