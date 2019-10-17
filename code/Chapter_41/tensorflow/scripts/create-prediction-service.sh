export MODEL_VERSION=v3
export MODEL_NAME=iris
export MODEL_BINARIES=$GCS_JOB_DIR/4/export/iris/1542258667
export REGION=us-central1

# Create a Cloud ML Engine model
echo "Creating model..."
gcloud ml-engine models create $MODEL_NAME --regions=$REGION

# Create a model version
echo "Creating model version..."
gcloud ml-engine versions create $MODEL_VERSION \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --runtime-version 1.8