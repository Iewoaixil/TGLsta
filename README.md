# TGLsta

## Installation

Before to execute TGLsta, it is necessary to install the following packages:

```shell
pip install -r requirements.txt
```
Additionally, you’ll need to install the Llama2 package separately from the official Llama2 website.

## Overall Structure

The project is organised as follows:

* `data/`contains the necessary dataset files;
* `pretrain.py`for pre-training the model;
* `test.py`for model evaluation and testing;
* `model.py`defines the pre-training model architecture and core logic;
* `model_g_coop.py`is model specifically designed for prompt tuning;
* `generate_llama2_embedding.py`to generate node feature embeddings using a pre-trained Llama2 model.

## Basic Usage

### Example

To generate node feature embeddings using a pre-trained Llama2 model, run:

```shell
python generate_llama2_embedding.py
```

To run pre-training:

```shell
python pretrain.py
```

To run testing:

```shell
python test.py
```

