# CAST: Cross-modal Alignment Similarity Test for Vision Language Models

## Abstract

Vision Language Models (VLMs) are typically evaluated with Visual Question Answering (VQA) tasks which assess a model's understanding of scenes. 
Good VQA performance is taken as evidence that the model will perform well on a broader range of tasks that require both visual and language inputs. 
However, scene-aware VQA does not fully capture input biases or assess hallucinations caused by a misalignment between modalities.
To address this, we propose a Cross-modal Alignment Similarity Test (CAST) to probe VLMs for **self-consistency** across modalities. 
This test involves asking the models to identify similarities between two scenes through text-only, image-only, or both and then assess the truthfulness of the similarities they generate. 
Since there is no ground-truth to compare against, this evaluation does not focus on objective accuracy but rather on whether VLMs are internally consistent in their outputs. 
We argue that while not all self-consistent models are capable or accurate, all capable VLMs must be self-consistent.



### Dataset

The dataset is sub-sampled from the [DOCCI](https://google.github.io/docci/) dataset (Onoe et al., 2024) and consists of 100 image pairs.

We select pairs of images that are visually similar and textually interesting. The similarity of the image pairs is determined by the cosine similarity of the image embeddings extracted from CLIP (Radford et al., 2021).

![CLIP similarity pairs](plots/clip_similarity.png)

![CLIP similarity vs Description length](plots/clip_vs_length.png)

### Requirements

The code is implemented in Python 3.12. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Run

To run CAST on our generated DOCCI subset, edit the `configs/eval.yaml` with desired configuration and run `python main.py`.
