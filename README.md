 <a name=top>

[<img width=100% src="https://github.com/hil-se/hil-se/blob/main/img/bar.png?raw=yes">](https://github.com/hil-se/hil-se/blob/main/README.md) 
&nbsp;&nbsp;&nbsp;[HOME](https://github.com/hil-se/hil-se#top) &nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;[NEWS](https://github.com/hil-se/hil-se/blob/main/subs/news.md#top) &nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;[PEOPLE](https://github.com/hil-se/hil-se/blob/main/people/people.md#top) &nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;[PROJECTS](https://github.com/hil-se/hil-se/blob/main/subs/projects.md#top) &nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;[PAPERS](https://github.com/hil-se/hil-se/blob/main/subs/papers.md#top) &nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;[RESOURCES](https://github.com/hil-se/hil-se/blob/main/subs/resources.md#top) &nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;[CONTACT-US](https://github.com/hil-se/hil-se/blob/main/subs/contact.md#top) &nbsp;&nbsp;&nbsp;


## Projects

### FairBalance: Mitigating Machine Learning Bias Against Multiple Protected Attributes With Data Balancing.

Aug 2021 | [repo](https://github.com/hil-se/FairBalance#top)

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Navigate to the source code:
```
cd src
```
2. RQ1: Can FairBalance mitigate bias against multiple protected attributes?
```
python main.py RQ1
```
3. RQ2: How does FairBalance perform comparing with the existing state-of-the-art bias mitigation algorithms?
```
python main.py RQ2
```
4. Run one single experiment with specific ML algorithm (e.g. Logistic Regression), dataset (e.g. Compas), bias mitigation algorithm (e.g. FairBalance), targeted attribute (e.g. None), and number of repeats (e.g. 1): 
```
python main.py one_exp LR compas FairBalance None 1
```

#### Acknowledgement
This work is built on [AIF360](https://github.com/Trusted-AI/AIF360). It is a great platform facilitating the creation and reproduction of AI bias mitigation algorithms.
