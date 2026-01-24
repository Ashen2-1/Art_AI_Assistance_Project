# Nexo Art VLM Fine-Tuning Lab

Fine-tuning open-source vision-language models (VLMs) to better understand artworks – styles, periods, and visual features – and integrating them into the Nexo art research platform.



## Overview

Most general VLMs are trained on broad internet images and **do not understand art history very well**.  
They often:

- Misclassify artistic styles and periods  
- Give shallow or generic descriptions of paintings  
- Ignore art-specific details (composition, technique, movement, symbolism)

For serious art students and researchers, this “generic” vision understanding is not enough.

**Nexo Art VLM Fine-Tuning Lab** aims to:

- Build an art-focused multimodal dataset (images + labels + descriptions)  
- Fine-tune open-source VLMs on this dataset  
- Serve the improved model as part of the broader **Nexo Art Research Platform**



## Goals

- Improve VLM performance on:
  - Art style classification (e.g., Baroque, Impressionism, Surrealism)
  - Period recognition (e.g., 19th-century European, post-war American)
  - Artwork description **from an art TA’s perspective**
- Provide a simple API so other Nexo projects can call:
  - “Describe this painting”
  - “What style/period is this likely from?”
  - “What research questions could I explore from this artwork?”



## High-level Approach

### 1. Dataset: Art-focused multimodal data

We plan to collect and curate:

- Public-domain artwork images (museums, open art datasets, etc.)
- Metadata & labels:
  - Artist, period, style, movement
  - Important visual features (composition, color, medium)
  - Short expert-style descriptions (where available)
