.PHONY: setup
setup:
	python data/preprocess.py --data_dir="data"

.PHONY: train
train:
	python main.py --epochs=1 --data_dir="data" --output_dir="output"

.PHONY: eval
eval:
	python main.py --do_eval --data_dir="data" --output_dir="output" --load_model="CIFARec-ml_1m"