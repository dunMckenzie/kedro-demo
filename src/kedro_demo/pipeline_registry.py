from kedro.pipeline import Pipeline 
from practices.pipeline import create_pipeline

def register_pipelines() -> dict:
    return {
        "de": create_pipeline()
    }