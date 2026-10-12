# Tile

## Description of Algorithim

## Problem

## How it Works

## Feasability Review

### User Need
The intended user of this Alzehimers Detection tool include:
- Doctors and practitioners to assist in medical diagnosis
- Researchers studying alzehimers, its features and predidiction / detection accuracy
- Researchers studying machine leanring and its application to medical datasets and image processing

prototype purpose, and 3–5
testable acceptance criteria addressing the domain’s open dilemma, including an operational/resource
constraint (e.g., peak VRAM, runtime).
### User Scope
The purpose of this ConvNeXt model is provide a prototype machine learning model to classify Alzhimers Disease. The model should flag ambiguous or low confidence classifications for review. Ambiguous cases are to be reliably indentified and have a predicted confidence that is correlated with diagnostic correctness.  
### Aceptance Criteria
The minimum acceptance criteria are as follows:
- Minimum classification test accuracy of 80% for Alzhimers Diagnosed brains versus Cognitivley Normal Brains.
- Has a bias towards flase positives over false negatives for miss calssifications
- Can identify ambiguous vs confidient classifications which correlate with diagnostic correctness
- Can extract discriminitive anatomical biomarkers despite ranges in head shape and age variation between patients within the latent space.
- Avoid patient data leakage via patient level seperation of medical images
- Avoid train, test and prediction data leackage
- Benchmark against the original "A ConvNet for the 2020s" paper [1].

### Model Choice

Model: ConvNeXt. 


Dataset: ADNI dataset
The model architecture will be as follows: 

The ADNI dataset provides structural MRI scans which can be preprocessed to include classifications of only Alzheimer’s disease or Cognitive Normal accesible via http://adni.loni.usc.edu. The data set has 591 sujects with Alzehimers Disease and 1479 cases of cognitivley normal casses. There is approximatley a 50/50 split in sex and a age split of (1552 70 -79, 1059 60-69, 539 80-89, 214 50-59 and 31 above 89).

The data will be split into 3 cases with each persons patient data strictly confined to a single split and not accrross cases to avoid data leakage :
- Train: %
- Test: %
- Predict: %

Difficulty Band: Hard 

### Course Concepts
how course concepts and literature support the choices.

### Preliminary Feasibility Evidence 
An initial data audit, baseline smoke-test experiment, or
measured resource estimate. A fully trained final model is not required.

### Risks
Key technical risks, computing budget, the next planned experiment,
and a workable fallback.
### Computing Budget
There are three compute options:
- RTX3080
- RTX2040
- Rangpur ....

The "A ConvNet for the 2020s" foundation paper used a V100 GPUs with 32 per-GPU batch size 
### Fallback

## Dependencies 

## REproducability

## Examples

### Inputs

### Outputs

### Visualisaions

# Open Research Dilema

## Quantitive Benchmarking

## Coputational Efficiency

## Qualitative Error

## Engineering Reccomendations

## References

[1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,”
in Proc. CVPR, 2022, pp. 11976–11986

