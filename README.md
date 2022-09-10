# TransPix2Pix
We propose a novel model for sketch-to-photo translation capable of  producing high fidelity realistic images with style adaptation. Inspired by the performance of the [pix2pix](https://arxiv.org/pdf/1611.07004.pdf) conditional GAN and [Vision Transformer](https://arxiv.org/pdf/2010.11929.pdf), we have developed a hybrid architecture, referred to as TransPix2Pix, that learns spatial and latent dependencies between convolutional features in the image-to-image translation tasks. To perform style adaptation the proposed model makes use of a separately pretrained style encoder, which converts image characteristics into an embedding vector. We apply the developed model for converting sketch images of cats into realistic photos and demonstrate that the proposed approach is more effective than the conventional one in terms of image quality.

![](examples/TransP2P_examples.jpg)

- **model_generator.py** and **model_discriminator.py** include TransPix2Pix Generator and Disrimiator models, respectively.
- **transformer_block.py** and **attention_block.py** include TransformerBlock and Self-AttentionBlock.
- **loss_network.py** - used for feature reconstruction loss computation.
- **model_ema.py** contains a class, that performs exponential mooving average of models weights.
- **StyleEncoder** directory include Style Encoder model and jupyter notebook with all the necessary training functions.
