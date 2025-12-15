import json
import os
import random
import pickle
from PIL import Image as PILImage
from datasets import Dataset, load_dataset

# 配置
SAMPLES_PER_ITEM = 24
NUM_GPUS = 8

final_output_dir = "/lustre/projects/med-multi-llm/haohan/cl_rif_dataset_v3"
checkpoint_dir = os.path.join(final_output_dir, "checkpoints")

print("Loading dataset from HuggingFace...")
dataset_dict = load_dataset("zhhxte/mllm_cl_clevr_processed")
train_data = list(dataset_dict["train"])
test_data = list(dataset_dict["test"])

print(f"\nOriginal dataset: {len(train_data)} train, {len(test_data)} test")

# 为训练集添加索引
for idx, item in enumerate(train_data):
    item['global_index'] = idx

# 从checkpoint加载所有GPU的结果
print("\nLoading results from checkpoints...")
all_results = {}

for gpu_id in range(NUM_GPUS):
    checkpoint_file = os.path.join(checkpoint_dir, f"gpu_{gpu_id}_checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                results = checkpoint_data['results']
                all_results[gpu_id] = results
                print(f"GPU {gpu_id}: Loaded {len(results)} results")
        except Exception as e:
            print(f"Failed to load GPU {gpu_id}: {e}")
    else:
        print(f"No checkpoint found for GPU {gpu_id}")

print(f"\nLoaded results from {len(all_results)}/{NUM_GPUS} GPUs")

# 合并结果
results = []
for gpu_id in range(NUM_GPUS):
    if gpu_id in all_results:
        results.extend(all_results[gpu_id])

# 按global_index排序以保持原始顺序
results.sort(key=lambda x: x.get('global_index', float('inf')))

print(f"Total processed items: {len(results)}")

if len(results) == 0:
    print("ERROR: No results found! Exiting...")
    exit(1)

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

print(f"\nSolvable (1 to {SAMPLES_PER_ITEM}/{SAMPLES_PER_ITEM} correct): {len(solvable)}")
print(f"Unsolvable (0/{SAMPLES_PER_ITEM} correct): {len(unsolvable)}")

# 计算max_sample
max_sample = min(len(solvable), len(unsolvable), 7500)
print(f"\nMax sample size: {max_sample}")

if max_sample == 0:
    print("ERROR: Not enough data to create datasets! Exiting...")
    exit(1)

# 随机采样
random.seed(42)
solvable_sampled = random.sample(solvable, max_sample) if len(solvable) >= max_sample else solvable
unsolvable_sampled = random.sample(unsolvable, max_sample) if len(unsolvable) >= max_sample else unsolvable

print(f"Sampled solvable: {len(solvable_sampled)}")
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

# 准备test集
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

# 准备random数据集
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

print("\nPreparing test dataset...")
test_dataset = prepare_test_dataset(test_data)
print(f"Test set: {len(test_dataset)} samples")

print("Preparing random train dataset...")
random_train_dataset = prepare_random_dataset(train_data, max_sample)
print(f"Random train set: {len(random_train_dataset)} samples")

# 保存三个数据集到本地
datasets_to_save = [
    ("mllm_cl_clevr_solvable", solvable_sampled),
    ("mllm_cl_clevr_unsolvable", unsolvable_sampled),
    ("mllm_cl_clevr_random", None)
]

for dataset_name, train_items in datasets_to_save:
    save_path = os.path.join(final_output_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nSaving to {save_path}...")
    
    if dataset_name == "mllm_cl_clevr_random":
        print(f"  Train: {len(random_train_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        random_train_dataset.to_parquet(os.path.join(save_path, "train.parquet"))
        test_dataset.to_parquet(os.path.join(save_path, "test.parquet"))
    else:
        if train_items:
            train_ds = convert_to_dataset(train_items, train_data)
            if train_ds:
                print(f"  Train: {len(train_ds)} samples")
                print(f"  Test: {len(test_dataset)} samples")
                
                train_ds.to_parquet(os.path.join(save_path, "train.parquet"))
                test_dataset.to_parquet(os.path.join(save_path, "test.parquet"))
    
    print(f"{dataset_name} saved successfully!")

print("\nAll datasets saved to local directory!")
print(f"Location: {final_output_dir}")
print(f"\nSummary:")
print(f"  mllm_cl_clevr_solvable: {len(solvable_sampled)} train samples")
print(f"  mllm_cl_clevr_unsolvable: {len(unsolvable_sampled)} train samples")
print(f"  mllm_cl_clevr_random: {len(random_train_dataset)} train samples")
print(f"  All datasets share the same test set: {len(test_dataset)} samples")
