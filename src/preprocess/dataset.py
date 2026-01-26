from typing import List, Dict, Tuple, Callable, Any
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BatchEncoding
from transformers.data.data_collator import default_data_collator
from collections import OrderedDict
import torch
import random
import pandas as pd



class RJPair2DSimplifiedDataset(Dataset):
    """
    encode data to the format of
    a dict resume/job would become: {
        "resume_sents": { # BatchEncoding of k sentences from a resume
            "input_ids": torch.tensor,
        },
        "job_sents": { # BatchEncoding of k sentences from a job
            "input_ids": torch.tensor,
        },
        "label": torch.tensor,  # shape batch_size
        ...
    }
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        resume_key_names: List[str],
        job_key_names: List[str],
        tokenizer_args: Dict[str, Any],
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[Tuple[str, str, int]],
        resume_taxon_token: str = "",
        job_taxon_token: str = "",
        query_prefix: str = "",
    ):
        print("Using query_prefix:", query_prefix)

        self.tokenizer = tokenizer
        self.query_prefix = query_prefix  # used by special encoders such as e5
        self.max_seq_len = max_seq_len
        self.resume_key_names = resume_key_names  # ensure the order of dict key is constant
        self.job_key_names = job_key_names
        self.resume_taxon_token = resume_taxon_token
        self.job_taxon_token = job_taxon_token
        self.tokenizer_args = tokenizer_args
        self.tokenizer_args["max_length"] = self.max_seq_len

        self.data = self.contruct_labeled_pairs(
            all_resume_dict, all_job_dict, label_pairs
        )
        self.encoded_data = self.encode_data(self.data)
        return

    def contruct_labeled_pairs(
        self,
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[dict],
    ):
        """
        label_pairs: [{'user_id': "xxx", 'jd_no': "xxx", 'satisifed': 0}, ...]
        output: [(resume_dict, job_dict, label), ...]
        """
        uid_to_resume = {}
        jd_no_to_job = {}

        for resume in all_resume_dict:
            resume_ = resume.copy()
            uid = str(resume_["user_id"])
            resume_.pop("user_id")
            uid_to_resume[uid] = resume_

        for job in all_job_dict:
            job_ = job.copy()
            jd_no = str(job_["jd_no"])
            job_.pop("jd_no")
            jd_no_to_job[jd_no] = job_

        # prepare data
        data = []
        for label_data in label_pairs:
            resume_id = str(label_data["user_id"])
            job_id = str(label_data["jd_no"])
            label = int(label_data["satisfied"])

            resume = uid_to_resume[resume_id]
            job = jd_no_to_job[job_id]
            data.append((resume, job, label))
        return data

    def _encode_single_dict(self, dict_data: Dict[str, str], type: str):
        keys_to_encode = self.resume_key_names if type == 'resume' else self.job_key_names
        keys_to_encode_set = set(keys_to_encode)
        for k, v in dict_data.items():
            assert(k in keys_to_encode_set)

        taxon_token = self.resume_taxon_token if type == 'resume' else self.job_taxon_token

        lines_to_encode = []
        for k in keys_to_encode:
            v = dict_data[k]
            content = f"{taxon_token}: section {k}. {v}"
            lines_to_encode.append(content)
        
        encoded_lines = self.tokenizer(lines_to_encode, **self.tokenizer_args)
        return encoded_lines

    def encode_data(
        self,
        data: List[Tuple[dict, dict, int]],
    ):
        """
        encode data to tensors
        """
        encoded_data = []
        for resume, job, label in tqdm(data, desc="Encoding data"):
            encoded_resume = self._encode_single_dict(resume, type='resume')
            encoded_job = self._encode_single_dict(job, type='job')

            encoded_data.append(
                {
                    "resume_sents": encoded_resume,
                    "job_sents": encoded_job,
                    "label": label,
                }
            )
        return encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index):
        return self.encoded_data[index]
