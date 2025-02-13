![Imgur](https://imgur.com/4RR3pjF.png)



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
  



## Step 5: Wrap Model in a Flask API

1. **Create `app.py`:**


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

## Deployment YAML File

The `deployment.yaml` file defines the deployment of your machine learning model on Kubernetes. This includes the number of replicas (pods) and the container image to use.

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

## Service YAML File

The `service.yaml` file is required to expose your deployed application and allow external access to the machine learning model running on Kubernetes.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Explanation

1. **Deployment YAML File**:
   - `replicas`: Specifies the number of pods to run for high availability.
   - `image`: The container image stored in Amazon ECR that contains your machine learning model API.
   - `containerPort`: The port the container listens on (e.g., 8000).

2. **Service YAML File**:
   - `kind: Service`: Defines the resource as a service.
   - `selector`: Links the service to the pods labeled as `app: ml-model`.
   - `ports`: Maps external traffic from port 80 to the container's port 8000.
   - `type: LoadBalancer`: Creates a load balancer to expose the service externally.

## Applying the Configuration

To deploy the `deployment.yaml` and `service.yaml` files to your Kubernetes cluster, use the following commands:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Verifying the Deployment

1. **Check the Deployment Status**:
   ```bash
   kubectl get deployments
   ```
2. **Check the Pods**:
   ```bash
   kubectl get pods
   ```
3. **Check the Service**:
   ```bash
   kubectl get services
   ```

The `EXTERNAL-IP` field in the service output will show the public IP address or DNS name for accessing your model's API.



4. **Expose the Service:**
   ```bash
   kubectl expose deployment ml-model-deployment --type=LoadBalancer --port=8000
   ```

5. **Get the External IP:**
   ```bash
   kubectl get services
   ```

---

## Step 9: Make Inferences

Use tools like `curl` or Postman to make POST requests to the API:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [[1.2, 3.4, 5.6]]}' <external-ip>:8000/predict
