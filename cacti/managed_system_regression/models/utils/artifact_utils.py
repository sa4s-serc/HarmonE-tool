import mlflow
import shutil

def log_knowledge_artifacts(knowledge_dir="knowledge"):
    mlflow.log_artifacts(knowledge_dir, artifact_path="harmone_knowledge")

def download_knowledge_artifacts(run_id, dst_path):
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(run_id, "harmone_knowledge", dst_path)
