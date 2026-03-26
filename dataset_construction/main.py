from pipeline import DatasetConstructionPipeline
from prompt import TEST_PROMPT

if __name__ == "__main__":

    pipeline = DatasetConstructionPipeline(model_id="Qwen/Qwen3.5-2B")
    result = pipeline._generate(TEST_PROMPT)
    print(result)