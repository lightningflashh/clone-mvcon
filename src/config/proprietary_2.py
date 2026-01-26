from typing import Callable

_max_seq_len_per_resume_feature = {
    "resume_summary": 128,
    "resume_experience": 512,
    "resume_skills": 256,
    "resume_education": 256,
    "resume_text": 512,
}


_max_seq_len_per_job_feature = {
    "jd_overview": 128,
    "jd_responsibilities": 512,
    "jd_requirements": 256,
    "jd_preferred": 256,
    "job_description_text": 512,
}

_max_key_seq_length = 16

def augment_proprietary_resume(
    resume: dict,
    augment_fn: Callable,
    data_type: str = 'positive'
) -> dict:
    """
    Augment các field của Resume.
    """
    assert data_type in ['positive', 'negative', 'noop', 'pretrain']
    
    # Copy để tránh ghi đè dữ liệu gốc
    augmented_resume = resume.copy()
    
    # Danh sách các trường text cần augment
    # Ở bản training thường (positive/negative), ta tập trung vào Exp và Summary
    fields_to_aug = ["resume_experience", "resume_summary"]
    
    if data_type == 'pretrain':
        # Nếu pretrain thì "vắt kiệt" hết các trường
        fields_to_aug += ["resume_skills", "resume_education", "resume_text"]

    for field in fields_to_aug:
        # Chỉ augment nếu field tồn tại và không rỗng
        if field in augmented_resume and augmented_resume[field]:
            val = augmented_resume[field]
            if isinstance(val, str) and len(val.strip()) > 0:
                augmented_resume[field] = augment_fn(val, data_type=data_type)
    
    return augmented_resume


def augment_proprietary_jd(
    jd: dict,
    augment_fn: Callable,
    data_type: str = 'positive'
) -> dict:
    """
    Augment các field của Job Description.
    """
    assert data_type in ['positive', 'negative', 'noop', 'pretrain']

    augmented_jd = jd.copy()
    
    # Các trường quan trọng của JD
    fields_to_aug = [
        "jd_responsibilities",
        "jd_requirements",
    ]
    
    if data_type == "pretrain":
        fields_to_aug += ["jd_overview", "jd_preferred", "job_description_text"]

    for field in fields_to_aug:
        if field in augmented_jd and augmented_jd[field]:
            val = augmented_jd[field]
            if isinstance(val, str) and len(val.strip()) > 0:
                augmented_jd[field] = augment_fn(val, data_type=data_type)
                
    return augmented_jd


# --- CONFIG TỔNG CHO MODEL ---
PROPRIETARY_CONFIG_2 = {
    "max_seq_len_per_feature": {
        **_max_seq_len_per_resume_feature,
        **_max_seq_len_per_job_feature,
    },
    "max_key_seq_length": _max_key_seq_length,
    "resume_taxon_token": "Resume Content",
    "job_taxon_token": "Job Information",
    
    # List các cột để DataLoader biết đường mà bốc dữ liệu
    "resume_key_names": list(_max_seq_len_per_resume_feature.keys()),
    "job_key_names": list(_max_seq_len_per_job_feature.keys()),
    
    # Mapping các hàm xử lý
    "resume_aug_fn": augment_proprietary_resume,
    "job_aug_fn": augment_proprietary_jd,
}