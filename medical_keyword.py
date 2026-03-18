import csv
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from datasets import Dataset

class MedicalKeywordDataset:
    def __init__(self):
        self.dataset = None

    def load_dataset_from_hf(self, hf_user_name='', data_set_name='symptom-to-keywords-dataset'):
        self.dataset = load_dataset(f"{hf_user_name}/{data_set_name}")
        return self.dataset

    def load_data_set_from_csv(self, file_path):
        self.dataset = pd.read_csv(file_path)
        return self.dataset

    def get_dataset(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")
        return self.dataset

    def make_data_items(self):
        for item in self.dataset:
            yield MedicalKeywordsDataItem(item["symptom"], item["keywords"])

    def make_hf_dataset(self):
        return Dataset.from_generator(self.make_data_items)

class MedicalKeywordsDataItem(BaseModel):
    symptom_description: str
    severity: str
    duration: str
    age_group: str
    keywords: list[str]
    prompt: Optional[str] = None

    def __init__(self, symptom_description: str, severity: str, duration: str, age_group: str, keywords: str):
        super().__init__(
            symptom_description=symptom_description,
            severity=severity,
            duration=duration,
            age_group=age_group,
            keywords=keywords.split(",")
        )
        self.prompt = self.make_prompt()
        self._symptom_token_count = None
        self._tokenizer = None

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    @property
    def symptomps_tokens_count(self):
        if self._symptom_token_count:
            return self._symptom_token_count
        return self._count_tokens(self.symptom_description)

    def to_dict(self):
        return {
            "symptom_description": self.symptom_description,
            "severity": self.severity,
            "duration": self.duration,
            "age_group": self.age_group,
            "keywords": self.keywords,
            "prompt": self.prompt
        }

    def __str__(self):
        return self.prompt
    
    def make_prompt(self):
        return f"Instruction: Extract the relevant medical keywords from the following symptoms and patient details.\
            \n\nInput:\nSymptoms: {self.symptom_description}\nSeverity: {self.severity}\n\
            Duration: {self.duration}\nAge Group: {self.age_group}\n\nOutput: {', '.join(self.keywords)}"

    def to_hf_format(self):
        return {
            "text": self.prompt
        }

    def to_hf_dataset(self):
        return Dataset.from_pandas(self.dataset)

    def _count_tokens(self, text):
        if not self._tokenizer:
            raise ValueError("Tokenizer not set. Please set the tokenizer first.")
        return len(self._tokenizer.encode(text, add_special_tokens=False))


    @classmethod
    def load_dataset_from_csv(cls, file_path, tokenizer):
        """Validates data items using pydantic and then returns all data items as MedicalKeywordsDataItem object"""
        data_items = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = cls.model_validate(row)
                item.set_tokenizer(tokenizer)
                data_items.append(item)
        return data_items

    @staticmethod
    def convert_items_to_hf_dataset(items):
        dataset = Dataset.from_list([item.to_hf_format() for item in items])
        return dataset

    @staticmethod
    def split_dataset(dataset, test_size=0.1):
        return dataset.train_test_split(test_size=test_size)

    @staticmethod
    def push_to_hub(dataset, hf_user_name, data_set_name, hf_token):
        dataset.push_to_hub(f"{hf_user_name}/{data_set_name}", token=hf_token)

    @staticmethod
    def split_and_push_items_to_hub(items, hf_user_name, data_set_name, hf_token, test_size=0.1):
        dataset = Dataset.from_list([item.to_hf_format() for item in items])
        split_dataset = MedicalKeywordsDataItem.split_dataset(dataset, test_size)
        MedicalKeywordsDataItem.push_to_hub(split_dataset, hf_user_name, data_set_name, hf_token)


def main():
    # medical_keyword_dataset = MedicalKeywordDataset()
    # medical_keyword_dataset.load_data_set_from_csv("synthetic_symptom_to_clinical_keywords_100k.csv")
    medical_keywords_items = MedicalKeywordsDataItem.load_dataset_from_csv("synthetic_symptom_to_clinical_keywords_100k.csv", None)
    print(medical_keywords_items[0])

if __name__ == "__main__":
    main() 