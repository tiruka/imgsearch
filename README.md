# Introduction of Deep Learning Image Search

You can try to search items by uploading an image file.

![Demo](https://raw.githubusercontent.com/tiruka/files/master/imgsearch/imageserach_demo.gif)

## MiddleWare

Backend: Keras
API: Flask

## Whole directory

```shell
imgsearch
├── README.md
├── app
│   ├── Dockerfile
│   ├── app.py
│   ├── etc
│   │   └── requirements.pip
│   ├── img_keras_cnn.h5
│   ├── index_label_mapping.json
│   ├── np_data.npy
│   ├── predict_img.py
│   ├── preprocess
│   │   ├── __init__.py
│   │   ├── cnn_train.py
│   │   ├── gen_data.py
│   │   └── predict.py
│   ├── settings.py
│   ├── static
│   │   ├── background
│   │   │   └── backgroundimage.jpeg
│   │   ├── css
│   │   │   └── style.css
│   │   ├── data
│   │   │   └── img
│   │   └── uploads
│   ├── templates
│   │   ├── base.html
│   │   ├── index.html
│   │   └── result.html
│   ├── train_model_batch.py
│   └── uwsgi.ini
├── docker-compose.yml
└── nginx
    ├── Dockerfile
    └── nginx.conf
```

## How to build model

### Deploy data

You should deploy data to make Keras to learn and build model.
In this application, you put directories including images (*.jpg) on `imgsearch/app/static/data/img`.

```shell
img
├── item1
│   ├── item1_1.jpg
│   ├── item1_2.jpg
│   └── item1_3.jpg
└── item2
│   ├── item2_1.jpg
│   ├── item2_2.jpg
│   ├── item2_3.jpg
│   └── item2_4.jpg
│
...
```

### Build model

At `imgsearch/app`, you should do the below commands

```python train_model_batch.py```

This commands make three files img_keras_cnn.h5, index_label_mapping.json, and np_data.npy.

- img_keras_cnn.h5: This file is a main file and model consisting of a compressed result of keras. When inferring, an application loads it.
- index_label_mapping.json: The inferred result by the model is expressed as numpy array and index. To specify what an item is from index, it finds form json file.
- np_data.npy: You do not use this file. This is made on the way of building model. This file has numpy data converted from image data.

### Customization of model

If you would like to enchane a model, there are some ways as below.

- Change the Keras model
- Tune and optimize parameters
- Amplify more and more images data

I do not explain it at detail, please explore it!

## How to serve application

### Preparation

Here I add explation to set up Amazon Linux2 (AWS EC2).

### Infrastructure

```shell
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker.service
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo systemctl restart docker.service
exit
(Login Again)
curl -L "https://github.com/docker/compose/releases/download/1.11.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### build

`docker-compose build (where docker-compose.yml file exists)`

### start

`docker-compose up -d`

### stop and clean

```shell
docker-compose stop
docker-compose rm
```

When you chagne and reflect flask app, you have to do it before rebuilding.

### miscellaneous

If necessary, you should add `sudo` head.
