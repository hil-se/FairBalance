# Relative Fairness Testing

#### Data (included in the [data/](./data/) folder)

 - [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).
   + [Ratings.csv](data/Ratings.csv) extracts ratings from the original data.

#### Pre-Trained weights

 - The VGG-16 model utilizes pre-trained weights on ImageNet data from [deepface_models](https://github.com/serengil/deepface_models).

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Create a folder checkpoint:
```
mkdir checkpoint
```
2. Download the pre-trained weights of VGG-16 model [vgg_face_weights.h5](https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5) and put it under _checkpoint/_
3. Navigate to the source code:
```
cd src
```
4. Generate results in [results/](results/)
```
python main.py
```

