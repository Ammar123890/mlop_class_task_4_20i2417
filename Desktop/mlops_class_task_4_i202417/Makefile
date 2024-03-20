.PHONY: setup install train run docker-build docker-run

setup:
	python3 -m venv ~/.wine-quality

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

train:
	python train_model.py

run:
	python app.py

docker-build:
	docker build -t wine-quality-app .

docker-run:
	docker run -p 5000:5000 wine-quality-app

all: install train
