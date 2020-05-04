# Youper - Data Challenge

### Please check the "Youper Data Challence Report" file to learn more about preprocessing and modelling for this challenge.

## Preparing an environment to run the code:

1. Install Anaconda or Miniconda Package Manager from [here](anaconda.com/products/individual)

2. Create a new virtual environment and install the required packages:
```bash
conda create -n journaling_and_reflections python
```
```bash
conda activate journaling_and_reflections
```
If using cuda:

```bash
conda install pytorch cudatoolkit=10.0 -c pytorch
```
3. Install all requiremnts
```bash
python reflection.py
```
## Working Example Demo - For quickly running the code

Click on [Working Example Demo](https://drive.google.com/file/d/1u4ZLaujaEDgyhdMd7_HXaytn7WFjjQHE/view?usp=sharing)

To run the model in your environment, just execute reflection.py file as shown in the demo
```bash
pip3 install -r requirements.txt (Python 3)
```
More about user inputsand outputs in the Demo:

Input1: general - I am doing bad/feeling sad etc.

Input2: Question/Journaling Entry

Input3: Context of Question / More information about the Journaling Entry

Final Output: Reflection for given Journaling Entry

## Notebooks:

I have explained my code in jupyter notebooks
Order of Notebooks:
1. Crawler.ipynb
2. Data Preprocessing and Visualization.ipynb
3. Baseline Model - Fine Tuning - Multi Label Multi Class Classification Model.ipynb
4. Contextual Similarity Model using BERT.ipynb

## Running all files from scratch
```bash
cd source
```
```bash
python crawler.py
```
```bash
python preprocessing.py
```
```bash
python baseline.py
```
```bash
python reflection.py
```
