export MODEL_VERSION=v1
export MODEL_NAME=iris_sklearn
export REGION=us-central1

# Create a Cloud ML Engine model
echo "Creating model..."
gcloud ml-engine models create $MODEL_NAME --regions=$REGION

# Create a model version
echo "Creating model version..."
gcloud ml-engine versions create $MODEL_VERSION \
    --model $MODEL_NAME \
    --config config.yaml