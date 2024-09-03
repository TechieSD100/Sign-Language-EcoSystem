# Sign-Language-EcoSystem
Run below commands for general usage:
```shell
cd "v3.0"
```
```shell
cd "Web Module"
```
Run the main module (Inference Classifier is merged into this):
```shell
python main.py
```
Open the below link in browser:
```shell
127.0.0.1:5000
```


## If you wish to train the models and customize sign languages and more, follow the below steps:
```shell
cd "v3.0"
```
```
cd "Web Module"
```

Edit collect_imgs.py and set the 'number_of_classes' variable to as many sign languages you wish to train.
Run the below command:
```shell
python collect_imgs.py
```
Press Q for recording each gesture.
The recorded video clips will be automatically turned into 100 frames for model training.

Run the below command to check if the gestures are properly detected:
```shell
python cv_test.py
```

Run the below command to create datasets:
```shell
python create_dataset.py
```
