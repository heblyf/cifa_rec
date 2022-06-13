.PHONY: setup
setup:
	python data/preprocess.py --data_dir="data"

.PHONY: train
train:
	python main.py --epochs=1

.PHONY: eval
eval:
	python main.py --do_eval
