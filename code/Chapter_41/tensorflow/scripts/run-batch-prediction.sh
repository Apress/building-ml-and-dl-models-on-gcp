export JOB_NAME=iris_prediction_6
export MODEL_NAME=iris
export MODEL_VERSION=v3
export TEST_FILE=gs://iris-dataset/hold_out_test.csv

# submit a batched job
gcloud ml-engine jobs submit prediction $JOB_NAME \
        --model $MODEL_NAME \
        --version $MODEL_VERSION \
        --data-format TEXT \
        --region $REGION \
        --input-paths $TEST_FILE \
        --output-path $GCS_JOB_DIR/predictions

# stream job logs
echo "Job logs..."
gcloud ml-engine jobs stream-logs $JOB_NAME

# read output summary
echo "Job output summary:"
gsutil cat $GCS_JOB_DIR/predictions/prediction.results-00000-of-00001