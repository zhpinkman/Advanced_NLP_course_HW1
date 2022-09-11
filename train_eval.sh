OUTPUT_FILE="results/stats_bpe.txt"
touch $OUTPUT_FILE

for DATASET in "products" "questions" "4dim" "odiya"
do

echo "RESULTS FOR: $DATASET NAIVE BAYES" >> $OUTPUT_FILE
python3 train.py \
    -m naivebayes \
    -i "datasets/custom/$DATASET.train.txt.train" \
    -o "nb.$DATASET.model" \
    >> $OUTPUT_FILE

echo "TRAIN" >> $OUTPUT_FILE
python3 classify.py \
    -m "nb.$DATASET.model" \
    -i "datasets/custom/$DATASET.train.txt.train" \
    -o "results/nb.$DATASET.train.preds" \
    >> $OUTPUT_FILE

echo "TEST" >> $OUTPUT_FILE
python3 classify.py \
    -m "nb.$DATASET.model" \
    -i "datasets/custom/$DATASET.train.txt.test" \
    -o "results/nb.$DATASET.train.preds" \
    >> $OUTPUT_FILE

done


for DATASET in "products" "questions" "4dim" "odiya"
do

echo "RESULTS FOR: $DATASET PERCEPTRON" >> $OUTPUT_FILE
python3 train.py \
    -m perceptron \
    -i "datasets/custom/$DATASET.train.txt.train" \
    -o "perceptron.$DATASET.model" \
    >> $OUTPUT_FILE

echo "TRAIN" >> $OUTPUT_FILE
python3 classify.py \
    -m "perceptron.$DATASET.model" \
    -i "datasets/custom/$DATASET.train.txt.train" \
    -o "results/perceptron.$DATASET.train.preds" \
    >> $OUTPUT_FILE

echo "TEST" >> $OUTPUT_FILE
python3 classify.py \
    -m "perceptron.$DATASET.model" \
    -i "datasets/custom/$DATASET.train.txt.test" \
    -o "results/perceptron.$DATASET.train.preds" \
    >> $OUTPUT_FILE

done

