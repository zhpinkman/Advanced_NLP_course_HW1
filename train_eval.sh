OUTPUT_FILE="results/stats_logistic_regression.txt"
rm -rf $OUTPUT_FILE 
touch $OUTPUT_FILE

for DATASET in "products" "questions" "4dim" "odiya"
do

echo "RESULTS FOR: $DATASET LOGISTIC REGRESSION" >> $OUTPUT_FILE
python3 train.py \
    -m logistic_regression \
    -i "datasets/custom/$DATASET.train.txt.train" \
    -o "lr.$DATASET.model" \
    --dev "datasets/custom/$DATASET.train.txt.test" \
    --devlabels "datasets/custom/$DATASET.train.txt.true" \
    --epochs 40


echo "TEST" >> $OUTPUT_FILE
python3 classify.py \
    -m "lr.$DATASET.model" \
    -i "datasets/custom/$DATASET.train.txt.test" \
    -o "results/lr.$DATASET.test.preds" \

python3 eval.py \
    -t "datasets/custom/$DATASET.train.txt.true" \
    -p "results/lr.$DATASET.test.preds" \
    >> $OUTPUT_FILE
done


# for DATASET in "products" "questions" "4dim" "odiya"
# do

# echo "RESULTS FOR: $DATASET NAIVE BAYES" >> $OUTPUT_FILE
# python3 train.py \
#     -m naivebayes \
#     -i "datasets/custom/$DATASET.train.txt.train" \
#     -o "nb.$DATASET.model"

# echo "TEST" >> $OUTPUT_FILE
# python3 classify.py \
#     -m "nb.$DATASET.model" \
#     -i "datasets/custom/$DATASET.train.txt.test" \
#     -o "results/nb.$DATASET.test.preds"

# python3 eval.py \
#     -t "datasets/custom/$DATASET.train.txt.true" \
#     -p "results/nb.$DATASET.test.preds" \
#     >> $OUTPUT_FILE

# done


# for DATASET in "products" "questions" "4dim" "odiya"
# do

# echo "RESULTS FOR: $DATASET PERCEPTRON" >> $OUTPUT_FILE
# python3 train.py \
#     -m perceptron \
#     -i "datasets/custom/$DATASET.train.txt.train" \
#     -o "perceptron.$DATASET.model" \
#     --dev "datasets/custom/$DATASET.train.txt.test" \
#     --devlabels "datasets/custom/$DATASET.train.txt.true" \
#     --epochs 40


# echo "TEST" >> $OUTPUT_FILE
# python3 classify.py \
#     -m "perceptron.$DATASET.model" \
#     -i "datasets/custom/$DATASET.train.txt.test" \
#     -o "results/perceptron.$DATASET.test.preds" \

# python3 eval.py \
#     -t "datasets/custom/$DATASET.train.txt.true" \
#     -p "results/perceptron.$DATASET.test.preds" \
#     >> $OUTPUT_FILE

# done

