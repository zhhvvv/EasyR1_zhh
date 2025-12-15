import json
import os
import random
import pickle
from datasets import Dataset, load_dataset

final_output_dir = "/lustre/projects/med-multi-llm/haohan/cl_rif_dataset_v2"
checkpoint_dir = os.path.join(final_output_dir, "checkpoints")

# 加载所有GPU的checkpoint结果
all_results = {}
for gpu_id in range(8):
    checkpoint_file = os.path.join(checkpoint_dir, f"gpu_{gpu_id}_checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
                all_results[gpu_id] = data['results']
                print(f"GPU {gpu_id}: Loaded {len(data['results'])} results")
        except Exception as e:
            print(f"GPU {gpu_id}: Failed to load - {e}")

# 合并所有结果
results = []
for gpu_id in sorted(all_results.keys()):
    results.extend(all_results[gpu_id])

# 按global_index排序
results.sort(key=lambda x: x.get('global_index', float('inf')))

print(f"\nTotal results recovered: {len(results)}")

# 加载原始数据集
print("Loading original dataset...")
dataset_dict = load_dataset("zhhxte/mllm_cl_textvqa")
train_data = list(dataset_dict["train"])
test_data = list(dataset_dict["test"])

# 分组：solvable vs unsolvable
solvable = []
unsolvable = []

for item in results:
    correct_count = item["correct_count"]
    total_count = item["total_count"]
    
    if total_count == 0 or correct_count == 0:
        unsolvable.append(item)
    else:
        solvable.append(item)

print(f"\nSolvable (1-8/8 correct): {len(solvable)}")
print(f"Unsolvable (0/8 correct): {len(unsolvable)}")

# 计算max_sample
max_sample = min(len(solvable), len(unsolvable), 7500)
print(f"Max sample size: {max_sample}")

# 随机采样
random.seed(42)
solvable_sampled = random.sample(solvable, min(max_sample, len(solvable)))
unsolvable_sampled = random.sample(unsolvable, min(max_sample, len(unsolvable)))

print(f"\nSampled solvable: {len(solvable_sampled)}")
print(f"Sampled unsolvable: {len(unsolvable_sampled)}")

# 转换为Dataset
def convert_to_dataset(items, original_data):
    if not items:
        return None
    
    data_dict = {
        "images": [],
        "problem": [],
        "answer": [],
        "correct_count": [],
        "total_count": [],
        "responses": []
    }
    
    for item in items:
        global_idx = item.get("global_index")
        if global_idx is not None and global_idx < len(original_data):
            images = original_data[global_idx].get("images")
            image = images[0] if images and len(images) > 0 else None
        else:
            image = None
        
        data_dict["images"].append([image] if image is not None else [])
        data_dict["problem"].append(item.get("problem", ""))
        data_dict["answer"].append(item.get("answer", ""))
        data_dict["correct_count"].append(item["correct_count"])
        data_dict["total_count"].append(item["total_count"])
        data_dict["responses"].append(json.dumps(item["responses"], ensure_ascii=False))
    
    return Dataset.from_dict(data_dict)

def prepare_test_dataset(test_data):
    data_dict = {
        "images": [],
        "problem": [],
        "answer": [],
        "correct_count": [],
        "total_count": [],
        "responses": []
    }
    
    for item in test_data:
        images = item.get("images")
        image = images[0] if images and len(images) > 0 else None
        
        data_dict["images"].append([image] if image is not None else [])
        data_dict["problem"].append(item.get("problem", ""))
        data_dict["answer"].append(item.get("answer", ""))
        data_dict["correct_count"].append(-1)
        data_dict["total_count"].append(-1)
        data_dict["responses"].append("[]")
    
    return Dataset.from_dict(data_dict)

def prepare_random_dataset(train_data, max_sample):
    random.seed(42)
    sampled_data = random.sample(train_data, min(max_sample, len(train_data)))
    
    data_dict = {
        "images": [],
        "problem": [],
        "answer": [],
        "correct_count": [],
        "total_count": [],
        "responses": []
    }
    
    for item in sampled_data:
        images = item.get("images")
        image = images[0] if images and len(images) > 0 else None
        
        data_dict["images"].append([image] if image is not None else [])
        data_dict["problem"].append(item.get("problem", ""))
        data_dict["answer"].append(item.get("answer", ""))
        data_dict["correct_count"].append(-1)
        data_dict["total_count"].append(-1)
        data_dict["responses"].append("[]")
    
    return Dataset.from_dict(data_dict)

test_dataset = prepare_test_dataset(test_data)
random_train_dataset = prepare_random_dataset(train_data, max_sample)

print(f"\nTest set: {len(test_dataset)} samples")
print(f"Random train set: {len(random_train_dataset)} samples")

# 保存三个数据集
datasets_to_save = [
    ("mllm_cl_textvqa_solvable", solvable_sampled),
    ("mllm_cl_textvqa_unsolvable", unsolvable_sampled),
    ("mllm_cl_textvqa_random", None)
]

for dataset_name, train_items in datasets_to_save:
    save_path = os.path.join(final_output_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nSaving to {save_path}...")
    
    if dataset_name == "mllm_cl_textvqa_random":
        print(f"  Train: {len(random_train_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        random_train_dataset.save_to_disk(os.path.join(save_path, "train"))
        test_dataset.save_to_disk(os.path.join(save_path, "test"))
    else:
        if train_items:
            train_ds = convert_to_dataset(train_items, train_data)
            if train_ds:
                print(f"  Train: {len(train_ds)} samples")
                print(f"  Test: {len(test_dataset)} samples")
                
                train_ds.save_to_disk(os.path.join(save_path, "train"))
                test_dataset.save_to_disk(os.path.join(save_path, "test"))
    
    print(f"{dataset_name} saved successfully!")

print("\nAll datasets saved!")
