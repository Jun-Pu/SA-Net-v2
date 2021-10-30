# SA-Net Version 2

------

# Introduction

In this work, we further improve the [SA-Net](https://github.com/PanoAsh/SA-Net) from the following aspects: 
 - We replace the ResNet blocks with hybrid-ViT based [transformer blocks](https://github.com/isl-org/DPT) at the AiF branch of the encoder.
 - We take advantage of the inter-slice features in both the encoding and decoding processes. Specifically, we discard the three 3D conv layers (with color gray in the figure1 of [SA-Net](https://github.com/PanoAsh/SA-Net)) at the FS branch of the encoder, and replace the original receptive field blocks (RFBs) with 3D RFBs. To fuse the 3D FS-based features and 2D AiF features at high-level, we further design a multi-head synergistic attention module (please refer to [codes](https://github.com/PanoAsh/SA-Net-v2/blob/main/models/NewBase_VIT.py)).

:running: :running: :running: ***KEEP UPDATING***.
