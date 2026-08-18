import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import pandas as pd
import json
from typing import List, Dict, Any
from src.jd_parser import partition_jd_description, parse_jd_to_tree

def clean_title(title: str) -> str:
    # Standardize common titles for easier matching
    t = title.lower().strip()
    if 'software engineer' in t or 'software developer' in t:
        return 'Software Engineer'
    if 'data scientist' in t:
        return 'Data Scientist'
    if 'product manager' in t:
        return 'Product Manager'
    if 'marketing manager' in t or 'marketing coordinator' in t:
        return 'Marketing Manager'
    if 'business analyst' in t:
        return 'Business Analyst'
    return title.strip()

def main():
    print("Building Job Descriptions (JD) Evaluation Dataset (Dataset-3)...")
    
    postings_path = "Data/dataset/linkedin_jd/postings.csv"
    job_ind_path = "Data/dataset/linkedin_jd/jobs/job_industries.csv"
    ind_map_path = "Data/dataset/linkedin_jd/mappings/industries.csv"
    
    if not all(os.path.exists(p) for p in [postings_path, job_ind_path, ind_map_path]):
        print("Error: Missing dataset CSV files in Data/dataset/linkedin_jd/")
        return
        
    postings = pd.read_csv(postings_path)
    job_ind = pd.read_csv(job_ind_path)
    ind_map = pd.read_csv(ind_map_path)
    
    # Merge to get industry name for each job posting
    df = postings.merge(job_ind, on="job_id").merge(ind_map, on="industry_id")
    
    # Industry mappings
    industry_groups = {
        "IT": ["Software Development", "IT Services and IT Consulting", "Technology, Information and Internet"],
        "Healthcare": ["Hospitals and Health Care", "Medical Practices"],
        "Finance": ["Financial Services", "Banking"],
        "Education": ["Higher Education", "Primary and Secondary Education"],
        "Marketing": ["Advertising Services", "Market Research"]
    }
    
    # Reverse mapping for quick lookup
    ind_to_group = {}
    for grp, inds in industry_groups.items():
        for ind in inds:
            ind_to_group[ind] = grp
            
    df = df[df["industry_name"].isin(ind_to_group.keys())].copy()
    df["industry_group"] = df["industry_name"].map(ind_to_group)
    
    # Drop duplicate job postings
    df = df.drop_duplicates(subset=["job_id"]).copy()
    
    print(f"Postings in target industries: {len(df)}")
    
    # Filter descriptions by length (word count between 150 and 1000)
    df["word_count"] = df["description"].fillna("").apply(lambda x: len(x.split()))
    df = df[(df["word_count"] >= 150) & (df["word_count"] <= 1000)].copy()
    print(f"Postings after length filter: {len(df)}")
    
    # Keep only records where company_name and description are not null
    df = df.dropna(subset=["company_name", "description"]).copy()
    
    # Validate section parsing - check that we can extract D2 and D3
    valid_jds = []
    for idx, row in df.iterrows():
        secs = partition_jd_description(row["description"])
        non_empty = sum(1 for k, v in secs.items() if len(v.strip()) > 30)
        if non_empty >= 2:
            valid_jds.append(row)
            
    valid_df = pd.DataFrame(valid_jds)
    print(f"Postings with valid domain sections (D2 & D3): {len(valid_df)}")
    
    if len(valid_df) < 150:
        print("Warning: Not enough valid job descriptions. Loosening criteria...")
        return
        
    # Standardize titles for negative pairing
    valid_df["clean_title"] = valid_df["title"].apply(clean_title)
    
    positive_pairs = []
    # 1. Generate Positive Pairs (same company, similar title or duplicate job_id postings)
    groups = valid_df.groupby(["company_name", "title"])
    for (company, title), group in groups:
        if len(group) >= 2:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    p1 = group.iloc[i]
                    p2 = group.iloc[j]
                    if p1["job_id"] != p2["job_id"] and p1["description"].strip() != p2["description"].strip():
                        positive_pairs.append((p1, p2, 1))
                        if len(positive_pairs) >= 66:
                            break
                if len(positive_pairs) >= 66:
                     break
        if len(positive_pairs) >= 66:
            break
            
    print(f"Generated natural positive pairs: {len(positive_pairs)}")
    
    # 2. Generate Hard Negatives (Topic Conflation: same standardised title but different industry)
    hard_negatives = []
    roles = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager", "Business Analyst"]
    
    for role in roles:
        role_df = valid_df[valid_df["clean_title"] == role]
        ind_groups = role_df.groupby("industry_group")
        for g1, grp1 in ind_groups:
            for g2, grp2 in ind_groups:
                if g1 == g2:
                    continue
                # Pair postings from different industries
                for i in range(min(len(grp1), 10)):
                    for j in range(min(len(grp2), 10)):
                        p1 = grp1.iloc[i]
                        p2 = grp2.iloc[j]
                        hard_negatives.append((p1, p2, 0))
                        if len(hard_negatives) >= 80:
                            break
                    if len(hard_negatives) >= 80:
                        break
                if len(hard_negatives) >= 80:
                    break
            if len(hard_negatives) >= 80:
                break
        if len(hard_negatives) >= 80:
            break
            
    print(f"Generated hard negative pairs (Topic Conflation): {len(hard_negatives)}")
    
    # 3. Generate Easy Negatives (different roles, different companies)
    easy_negatives = []
    for i in range(len(valid_df)):
        p1 = valid_df.iloc[i]
        for j in range(i+1, len(valid_df)):
            p2 = valid_df.iloc[j]
            if p1["company_name"] != p2["company_name"] and p1["clean_title"] != p2["clean_title"]:
                easy_negatives.append((p1, p2, 0))
                if len(easy_negatives) >= 54:
                    break
        if len(easy_negatives) >= 54:
            break
            
    print(f"Generated easy negative pairs: {len(easy_negatives)}")
    
    # Combine and shuffle
    all_pairs = positive_pairs + hard_negatives + easy_negatives
    print(f"Total pairs generated: {len(all_pairs)}")
    
    # Output to pairs.csv
    pairs_list = []
    used_job_ids = set()
    for idx, (p1, p2, label) in enumerate(all_pairs):
        pairs_list.append({
            "pair_id": f"JD_PAIR_{idx:03d}",
            "job_id_A": p1["job_id"],
            "title_A": p1["title"],
            "company_A": p1["company_name"],
            "industry_A": p1["industry_group"],
            "job_id_B": p2["job_id"],
            "title_B": p2["title"],
            "company_B": p2["company_name"],
            "industry_B": p2["industry_group"],
            "label": label
        })
        used_job_ids.add(p1["job_id"])
        used_job_ids.add(p2["job_id"])
        
    pairs_df = pd.DataFrame(pairs_list)
    os.makedirs("Data/dataset/linkedin_jd", exist_ok=True)
    pairs_df.to_csv("Data/dataset/linkedin_jd/pairs.csv", index=False)
    print("Saved pairs to Data/dataset/linkedin_jd/pairs.csv")
    
    # Parse trees for all selected job descriptions
    selected_postings = valid_df[valid_df["job_id"].isin(used_job_ids)]
    trees_unnormalized = {}
    
    for idx, row in selected_postings.iterrows():
        tree = parse_jd_to_tree(row["job_id"], row["title"], row["company_name"], row["description"])
        trees_unnormalized[str(row["job_id"])] = tree.to_dict()
        
    with open("Data/dataset/linkedin_jd/trees_unnormalized.json", "w", encoding="utf-8") as f:
        json.dump(trees_unnormalized, f, ensure_ascii=False, indent=2)
    print("Saved unnormalized trees to Data/dataset/linkedin_jd/trees_unnormalized.json")

if __name__ == "__main__":
    main()
