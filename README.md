### FairBalance: Improving Machine Learning Fairness on Multiple Sensitive Attributes With Data Balancing.

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Navigate to the source code:
```
cd src
```
2. RQ1+2: Can FairBalance (and FairBalanceClass) mitigate bias against multiple Sensitive attributes?
```
python main.py RQ1
```
3. RQ3: How does FairBalance perform comparing with the existing state-of-the-art bias mitigation algorithms?
```
python main.py RQ3
```
4. Run one single experiment with specific ML algorithm (e.g. Logistic Regression), dataset (e.g. Compas), bias mitigation algorithm (e.g. FairBalance), target Sensitive attribute (e.g. None), and number of repeats (e.g. 1): 
```
python main.py one_exp LR compas FairBalance None 1
```

#### Acknowledgement
This work is built on [AIF360](https://github.com/Trusted-AI/AIF360). The [aif360](https://github.com/hil-se/FairBalance/tree/main/aif360) folder is directed cloned from the [AIF360](https://github.com/Trusted-AI/AIF360) repo on May 1st 2021. It is a great platform facilitating the creation and reproduction of AI bias mitigation algorithms.
