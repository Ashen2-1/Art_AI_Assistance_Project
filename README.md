# NEXO Multimodal AI / Vision-Language System

Multimodal AI and vision-language-model development for **NEXO**, a full-stack AI research platform designed for reasoning across documents, images, and research sources.

This repository focuses on the AI subsystem responsible for:

- Image understanding
- Multimodal question answering
- Vision-language inference
- Image-grounded research assistance
- Future domain adaptation using public-domain artwork datasets

The current system integrates **LLaVA-1.5-7B** using PyTorch and Hugging Face Transformers, with GPU inference and memory optimization for multimodal research workflows.

---

## Project Context

NEXO is a multimodal research platform that brings together:

- Research documents
- Images
- Notes
- Academic sources
- Retrieval-Augmented Generation
- Vision-language models
- Interactive AI research tools

The project advanced to the **Hult Prize national round**.

This repository contains the multimodal AI / vision-language component of the broader NEXO system.

Live platform:

[NEXO Research Platform](https://nexo-art-research.vercel.app/)

---

## System Overview

The larger NEXO architecture combines document retrieval and multimodal reasoning.

```text
Research Materials
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Documents            Images
        │               │
        ▼               ▼
 OCR / Parsing     Image Processing
        │               │
        ▼               │
 Text Chunking          │
        │               │
        ▼               │
 Embeddings             │
        │               │
        ▼               │
Vector Retrieval        │
        │               │
        └───────┬───────┘
                ▼
        Query / Context
                │
        ┌───────┴────────┐
        ▼                ▼
       RAG        Vision-Language Model
        │                │
        └───────┬────────┘
                ▼
      Grounded Multimodal Response
```

This repository focuses primarily on the **vision-language-model branch** of the architecture.

---

## Current Vision-Language Model

The current implementation uses:

- **LLaVA-1.5-7B**
- PyTorch
- Hugging Face Transformers
- CUDA GPU inference
- 4-bit quantization
- FP16 computation

The model accepts an image together with a natural-language question or research prompt.

```text
Image
  +
Research Question
        │
        ▼
Image Preprocessing
        │
        ▼
Multimodal Prompt Construction
        │
        ▼
LLaVA-1.5-7B
        │
        ▼
Generated Multimodal Response
```

---

## Multimodal Question Answering

The system allows users to ask natural-language questions about visual materials.

Potential use cases include:

- Artwork interpretation
- Historical-image analysis
- Visual evidence examination
- Image-grounded research questions
- Comparing visual and textual sources
- Identifying visual characteristics
- Supporting research involving mixed media

Example:

```text
User uploads artwork
        │
        ▼
"What visual characteristics are important in this image?"
        │
        ▼
Vision-Language Model
        │
        ▼
Image-grounded response
```

The system is intended to assist researchers rather than replace human interpretation.

---

## Model Integration

The vision-language model is integrated using PyTorch and Hugging Face Transformers.

The implementation includes:

- Model initialization
- Processor initialization
- Image preprocessing
- Prompt construction
- Image + text input handling
- GPU inference
- Response generation
- Configurable generation parameters

Generation behavior can be controlled using parameters such as:

- Temperature
- Top-p sampling
- Maximum generated tokens
- Repetition penalty

---

## GPU and Memory Optimization

Running a 7B-parameter multimodal model can require significant GPU memory.

To reduce inference memory requirements, the system currently uses:

- **4-bit quantization**
- FP16 computation
- GPU execution
- PyTorch inference mode

Conceptually:

```text
Full-Precision Model
        │
        ▼
4-bit Quantized Weights
        │
        +
FP16 Computation
        │
        ▼
Reduced GPU Memory Usage
        │
        ▼
Multimodal Inference
```

This makes local or limited-resource experimentation more practical.

---

## Integration With NEXO

The vision-language subsystem operates as part of the larger NEXO backend.

```text
React Frontend
      │
      ▼
FastAPI Backend
      │
      ▼
Query Router
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
Document Query        Multimodal Query
      │                      │
      ▼                      ▼
RAG Pipeline          LLaVA-1.5-7B
      │                      │
      └──────────┬───────────┘
                 ▼
             Response
```

The broader NEXO platform also includes:

- React frontend
- FastAPI backend
- PostgreSQL
- PDF ingestion
- OCR
- Text chunking
- Embeddings
- Vector similarity retrieval
- Source-grounded RAG
- Research-source discovery
- Interactive research workspace

---

## Technology Stack

### AI / Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- LLaVA-1.5-7B
- Vision-Language Models
- Multimodal inference
- 4-bit quantization
- FP16 inference

### Backend / Integration

- FastAPI
- REST APIs
- PostgreSQL

### Broader NEXO Platform

- React
- RAG
- OCR
- Embeddings
- Vector retrieval
- Research document processing

---

## Current Capabilities

Current implemented capabilities include:

- Image + text multimodal input
- Vision-language inference
- Image-question answering
- GPU-based model execution
- Quantized model loading
- Configurable response generation
- Integration with NEXO query workflows
- Multimodal research assistance

---

# Planned Artwork Dataset Pipeline

A major planned extension is the construction of an artwork-focused multimodal dataset using **The Metropolitan Museum of Art Open Access collection**.

The Met provides public artwork metadata and public-domain image resources that can be accessed programmatically.

The intended pipeline is:

```text
The Met Open Access API
        │
        ▼
Artwork Object IDs
        │
        ▼
Artwork Metadata
        +
Public-Domain Images
        │
        ▼
Filtering / Cleaning
        │
        ▼
Structured Artwork Dataset
        │
        ▼
Image–Text Training Examples
        │
        ▼
VLM Fine-Tuning / Adaptation
        │
        ▼
Art-Focused Evaluation
```

---

## The Met API Integration

The planned data-ingestion workflow will use The Met Open Access API to collect artwork information programmatically.

Potential fields include:

- Object ID
- Title
- Artist
- Artist nationality
- Object date
- Period
- Dynasty
- Culture
- Medium
- Dimensions
- Department
- Classification
- Object name
- Tags
- Primary image
- Public-domain status

Example conceptual request:

```text
Search API
    │
    ▼
Artwork Object IDs
    │
    ▼
Object API
    │
    ▼
Metadata + Image URL
```

Only appropriate public-domain images and metadata will be used for dataset construction.

---

## Dataset Construction

Raw museum metadata is not automatically suitable for model training.

The planned dataset builder will perform:

1. API data collection
2. Public-domain filtering
3. Missing-field removal
4. Metadata normalization
5. Image downloading
6. Duplicate detection
7. Text cleaning
8. Training-example generation
9. Train / validation / test splitting

Target structure:

```text
Artwork Image
      +
Structured Metadata
      +
Research Question
      +
Grounded Answer
```

---

## Example Training Record

A possible multimodal instruction example could be:

```json
{
  "image": "artwork_12345.jpg",
  "question": "Describe the visual characteristics of this artwork and identify relevant contextual information.",
  "answer": "The artwork uses ... According to the museum metadata, it was created by ... during ..."
}
```

Another possible task:

```json
{
  "image": "artwork_67890.jpg",
  "question": "What medium was used to create this artwork?",
  "answer": "Oil on canvas."
}
```

More advanced examples could combine visual reasoning and structured museum metadata.

---

## Planned Training Tasks

Potential fine-tuning or adaptation tasks include:

- Artwork description
- Image-grounded question answering
- Artist identification
- Period recognition
- Medium recognition
- Visual feature analysis
- Art-historical context generation
- Comparison between artworks
- Image + metadata reasoning
- Research-oriented multimodal QA

---

## Planned Fine-Tuning Pipeline

The planned model-development workflow is:

```text
The Met API
      │
      ▼
Dataset Builder
      │
      ▼
Image–Text / Instruction Dataset
      │
      ▼
Base Vision-Language Model
      │
      ▼
Fine-Tuning / Parameter-Efficient Adaptation
      │
      ▼
Art-Focused VLM
      │
      ▼
Evaluation
```

Possible approaches to investigate include:

- LoRA
- QLoRA
- Instruction tuning
- Parameter-efficient fine-tuning
- Domain adaptation

The specific training strategy will be selected after dataset quality and available compute resources are evaluated.

---

## Why Fine-Tune?

The current LLaVA model provides strong general multimodal capabilities.

However, general-purpose models may have limitations when working with specialized research domains.

Domain adaptation could potentially improve:

- Art-specific vocabulary
- Artwork description quality
- Museum metadata reasoning
- Period / medium recognition
- Visual feature interpretation
- Domain-specific question answering

The objective is not simply to memorize artwork metadata.

The longer-term goal is to improve the model's ability to reason about visual research materials while remaining grounded in reliable source information.

---

## Evaluation Plan

Model development will require evaluation before and after adaptation.

Potential evaluation dimensions include:

### Factual Accuracy

Does the generated response correctly reflect known artwork metadata?

### Visual Grounding

Does the response describe features actually visible in the image?

### Hallucination

Does the system invent artists, dates, media, or historical details?

### Research Usefulness

Does the response provide information that is useful in a research workflow?

### Generalization

Can the model answer questions about artworks not present in the training set?

---

## Example Evaluation Pipeline

```text
Test Artwork
      │
      +
Research Question
      │
      ▼
Base Model
      │
      ▼
Response A

Test Artwork
      │
      +
Same Question
      │
      ▼
Adapted Model
      │
      ▼
Response B

        │
        ▼
Compare:
- Accuracy
- Grounding
- Hallucination
- Relevance
```

---

## Connection to RAG

Fine-tuning and RAG solve different problems.

The planned NEXO architecture may ultimately combine both.

```text
Domain-Adapted VLM
        +
Retrieved Research Sources
        │
        ▼
Multimodal Grounded Generation
```

Fine-tuning may improve domain understanding and response behavior.

RAG provides access to external source material and citations.

Combining the two could provide stronger research-oriented multimodal responses than either approach alone.

---

## Future Multimodal Retrieval

Another planned direction is image-aware retrieval.

Current document retrieval primarily focuses on text embeddings.

A future system could support:

```text
Artwork Image
      │
      ▼
Multimodal Embedding
      │
      ▼
Vector Search
      │
      ▼
Visually / Semantically Similar Artworks
```

This could enable:

- Similar-artwork discovery
- Visual comparison
- Cross-collection research
- Image-to-source retrieval
- Multimodal recommendation

---

## Planned Research Workflow

The longer-term NEXO workflow could become:

```text
Researcher Uploads Image
        │
        ▼
Multimodal Model Analysis
        │
        ▼
Museum / Research Source Retrieval
        │
        ▼
Relevant Documents + Artworks
        │
        ▼
Grounded Multimodal Reasoning
        │
        ▼
Research Response
```

---

## Current Status

**Active development — 2026**

### Implemented

- LLaVA-1.5-7B integration
- PyTorch inference
- Hugging Face Transformers
- Image + text input
- Multimodal question answering
- GPU inference
- 4-bit quantization
- FP16 computation
- NEXO backend integration

### In Development / Planned

- The Met Open Access API integration
- Artwork dataset builder
- Automated metadata cleaning
- Image dataset collection
- Multimodal instruction dataset generation
- Fine-tuning experiments
- LoRA / QLoRA evaluation
- Art-domain benchmarking
- Multimodal retrieval
- Improved grounding
- Model comparison and evaluation

---

## Future Development

Planned development includes:

- The Met API dataset pipeline
- Public-domain artwork collection
- Automated dataset generation
- Artwork-focused VLM adaptation
- Model benchmarking
- LoRA / QLoRA experiments
- Better multimodal prompting
- Image / text context integration
- Image-to-document cross-referencing
- Multimodal retrieval
- Structured AI outputs
- Improved grounding
- Reduced hallucination
- More efficient GPU inference

---

## Engineering Goals

The goal of this project is not only to experiment with AI models.

The broader engineering objective is to connect:

```text
Public Research Data
        +
Machine Learning
        +
Retrieval Systems
        +
Full-Stack Software
        │
        ▼
Usable Multimodal Research Platform
```

The system is being developed as an integrated AI product rather than a standalone model demo.

---

## Related Project

This repository is part of the larger **NEXO multimodal research platform**.

Live application:

[NEXO Research Platform](https://nexo-art-research.vercel.app/)

Main areas of the broader system include:

- RAG
- Document processing
- Academic-source retrieval
- Multimodal AI
- Research organization
- Interactive AI tools

---

## Author

**Tom Li**  
Computer Engineering — University of Waterloo

Areas of interest:

- Multimodal AI
- Vision-Language Models
- RAG
- Computer Vision
- Machine Learning
- AI Systems
- Full-Stack AI Applications

GitHub: [Ashen2-1](https://github.com/Ashen2-1)
