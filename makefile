.PHONY: clean
clean:
	rm -rf output/CIFARec-*
	rm -rf data/seq_dict.pkl

.PHONY: format
format:
	black --line-length 120 .
