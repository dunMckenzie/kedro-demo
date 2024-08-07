from kedro.pipeline import node, Pipeline
from practices.nodes import load_data, preprocess_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_data,
                inputs="params:file_path",
                outputs="raw_data",
                name="load_data_node"
            ),
            node(
                func=preprocess_data,
                inputs="raw_data",
                outputs=["x", "y", "cv"],
                name="preprocess_data_node",
            ),
            node(
                func = train_model,
                inputs=["x", "y"],
                outputs=["model", "xtrain", "xtest", "ytrain", "ytest"],
                name="evaluate_model_node"
            )
        ]
    )

