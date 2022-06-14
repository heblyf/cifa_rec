.PHONY: setup
setup:
	python data/preprocess.py --data_dir="data" --limit=300000

.PHONY: train
train:
	python main.py --data_dir="data" --output_dir="output"

.PHONY: eval
eval:
	python main.py --do_eval --data_dir="data" --output_dir="output" --load_model="CIFARec-book_recommend"