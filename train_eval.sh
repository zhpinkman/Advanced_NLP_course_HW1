for DATASET in "products" "questions" "4dim" "odiya"
do
python3 train.py \
    -m naivebayes \
    -i "datasets/custom/$DATASET.train.txt.train" \
    -o "nb.$DATASET.model"

python3 classify.py \
    -m "nb.$DATASET.model" \
    -i "datasets/custom/$DATASET.train.txt.train" \
    -o "results/nb.$DATASET.train.preds"

python3 classify.py \
    -m "nb.$DATASET.model" \
    -i "datasets/custom/$DATASET.train.txt.test" \
    -o "results/nb.$DATASET.train.preds"

done

