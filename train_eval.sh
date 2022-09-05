python3 train.py \
    -m naivebayes \
    -i datasets/products.train.txt \
    -o nb.products.model


python3 classify.py \
    -m nb.products.model \
    -i datasets/products.train.txt \
    -o results/nb.products.train.preds