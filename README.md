# Salient Co-Speech Gesture Generation

A repository for generating salient co-speech gesture. This is the repository for paper *Salient Co-Speech Gesture Synthesizing with Discrete Motion Representation*.

## Quick Start

### Installation

```shell
conda create -f environment.yml
```

### Run

1. Run `train_vae.py` to train **Motion Auto-Encoder**.
2. Run `train_transformer.py` to train **Speech-Gesture Mapping** network.
3. Run `evaluate_transformer.py` to evaluate FGD and MVD.
4. Run `inference_transformer.py` to evaluate SMS.

## Data

We leverage `lmdb` to store the dataset. The data can be downloaed at link.

## Project Overview

TODO
