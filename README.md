# Machine-learning-Deployment-on-AWS
Containerization of ML model for KUBERNETES EKS deployment on AWS
# Machine Learning Model Deployment to AWS EKS

This guide outlines the process of setting up the AWS CLI, managing IAM privileges, creating a virtual environment, training a machine learning model, saving it in ONNX format, wrapping it in an API, containerizing with Docker, pushing to Amazon ECR, and deploying to Kubernetes using AWS EKS.

---

## Prerequisites

- **Ubuntu OS**
- Python (>=3.8)
- Docker installed and running
- AWS account and administrative privileges
- AWS CLI installed
- kubectl and eksctl installed
- Visual Studio Code (VSCode) installed

---

## Step 1: Install and Configure AWS CLI

1. **Install AWS CLI:**
   ```bash
   sudo apt update
   sudo apt install awscli -y
   ```

2. **Verify Installation:**
   ```bash
   aws --version
   ```

3. **Configure AWS CLI:**
   ```bash
   aws configure
   ```
   Provide the following details:
   - Access Key ID
   - Secret Access Key
   - Default region
   - Output format (e.g., json)

---

## Step 2: Set Up IAM Privileges

Ensure the IAM user/role has the following permissions:

- **AmazonEKSClusterPolicy**
- **AmazonEKSWorkerNodePolicy**
- **AmazonEKS_CNI_Policy**
- **AmazonEC2ContainerRegistryFullAccess**

If permissions are missing:

1. Ask an administrator to attach policies:
   ```bash
   aws iam attach-user-policy --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy --user-name <your-iam-username>
   ```

2. Verify attached policies:
   ```bash
   aws iam list-attached-user-policies --user-name <your-iam-username>
   ```

---

## Step 3: Set Up Python Environment

1. **Create a Project Directory:**
   ```bash
   mkdir ml_kubernetes
   cd ml_kubernetes
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Libraries:**
   Create a `requirements.txt` file with the following content:
   ```
   flask
   numpy
   onnx
   boto3
   joblib
   ```
   Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 4: Train and Save the Machine Learning Model

1. **Train the Model:**
   Write your training script (e.g., `train.py`) to preprocess data and train the model.

2. **Save the Model in ONNX Format:**
   ```python
   import onnx
   from sklearn.linear_model import LogisticRegression
   from skl2onnx import convert_sklearn
   from skl2onnx.common.data_types import FloatTensorType

   # Example: Train Logistic Regression and save as ONNX
   model = LogisticRegression()
   model.fit(X_train, y_train)

   initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
   onnx_model = convert_sklearn(model, initial_types=initial_type)
   with open("model.onnx", "wb") as f:
       f.write(onnx_model.SerializeToString())
   ```

---

## Step 5: Wrap Model in a Flask API

1. **Create `app.py`:**
   ```python
   from flask import Flask, request, jsonify
   import onnxruntime as rt
   import numpy as np

   app = Flask(__name__)

   session = rt.InferenceSession("model.onnx")
   input_name = session.get_inputs()[0].name

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json['data']
       np_data = np.array(data, dtype=np.float32)
       result = session.run(None, {input_name: np_data})
       return jsonify(result=result[0].tolist())

   if __name__ == "__main__":
       app.run(host="0.0.0.0", port=8000)
   ```

2. **Test Locally:**
   ```bash
   python app.py
   ```

---

## Step 6: Containerize with Docker

1. **Create a `Dockerfile`:**
   ```dockerfile
   FROM python:3.8-slim

   WORKDIR /app

   COPY requirements.txt requirements.txt
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   CMD ["python", "app.py"]
   ```

2. **Build the Docker Image:**
   ```bash
   docker build -t ml-model-api .
   ```

3. **Run the Docker Container Locally:**
   ```bash
   docker run -p 8000:8000 ml-model-api
   ```

---

## Step 7: Push to Amazon ECR

1. **Create an ECR Repository:**
   ```bash
   aws ecr create-repository --repository-name ml-model-api
   ```

2. **Authenticate Docker to ECR:**
   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
   ```

3. **Tag and Push Image:**
   ```bash
   docker tag ml-model-api:latest <account_id>.dkr.ecr.<region>.amazonaws.com/ml-model-api:latest
   docker push <account_id>.dkr.ecr.<region>.amazonaws.com/ml-model-api:latest
   ```

---

## Step 8: Deploy to Kubernetes Using EKS

1. **Create an EKS Cluster:**
   ```bash
   eksctl create cluster --name ml-cluster --region <region> --nodes 2
   ```

2. **Update kubeconfig:**
   ```bash
   aws eks --region <region> update-kubeconfig --name ml-cluster
   ```

3. **Create a Deployment YAML File:**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: ml-model-deployment
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: ml-model
     template:
       metadata:
         labels:
           app: ml-model
       spec:
         containers:
         - name: ml-model
           image: <account_id>.dkr.ecr.<region>.amazonaws.com/ml-model-api:latest
           ports:
           - containerPort: 8000
   ```

4. **Apply the Deployment:**
   ```bash
   kubectl apply -f deployment.yaml
   ```

5. **Expose the Service:**
   ```bash
   kubectl expose deployment ml-model-deployment --type=LoadBalancer --port=8000
   ```

6. **Get the External IP:**
   ```bash
   kubectl get services
   ```

---

## Step 9: Make Inferences

Use tools like `curl` or Postman to make POST requests to the API:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [[1.2, 3.4, 5.6]]}' <external-ip>:8000/predict
