# CURTIS - CURe by Therapy Intelligent System

### Curtis is an emotional health therapy bot which comforts users with short, accurate, and empathetic responses using deep contextual NLP models.

## Preparing an environment to run the code:

1. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/products/individual)
2. Create a new virtual environment and install the required packages:
```bash
conda create -n CURTIS_reflections python
```
```bash
conda activate CURTIS_reflections
```
If using cuda:
```bash
conda install pytorch cudatoolkit=10.0 -c pytorch
```
3. Install all requirements
```bash
pip install -r requirements.txt 
```
## Quick Example 

![Image description](https://github.com/kaushikData/Reflection-Predictor-Youper/blob/master/demo/Demo-Example.png)

### To run the model in your environment, just execute reflection.py file as shown in the demo
```bash
python source/reflection.py
```
More about user inputsand outputs in the Demo:

Input1: general - example: I am doing bad/feeling sad, etc.
Input2: Question/Problem
Input3: Context of Question / More information about the problem
Final Output: Reflection / Short Response

## Notebooks:

I have explained my code in jupyter notebooks

Order of Notebooks:
1. Crawler.ipynb
2. Data Preprocessing.ipynb
3. Baseline Model - Fine Tuning - Multi-Label Multi-Class Classification Model.ipynb
4. Contextual Similarity Model using BERT.ipynb

## Running all files from scratch

Change the present working directory to source
```bash
cd source
```
Run crawler.py to crawl all the questions and answers from CouncilChat website

```bash
python crawler.py
```
Run preprocessing.py to generate reflections in the training data
```bash
python preprocessing.py
```
Run baseline.py to train and test the baseline chatbot
```bash
python baseline.py
```
Run contextual_similarity.py to train the current model and save contextual vectors to datafolder.

```bash
python contextual_similarity.py
```
Run reflections.py to test the chatbot
```bash
python reflection.py
```
### Interface (In Progress)

[![Click here to view the demo of interface](https://www.youtube.com/watch?v=CTroNoxClQs)](https://www.youtube.com/watch?v=CTroNoxClQs)