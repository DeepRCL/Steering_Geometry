from pipeline import DatasetConstructionPipeline
from prompt import TEST_PROMPT

if __name__ == "__main__":

    pipeline = DatasetConstructionPipeline()
    result = pipeline.generate(TEST_PROMPT)
    print(result)