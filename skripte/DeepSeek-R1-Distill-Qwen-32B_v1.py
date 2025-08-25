import os
import datetime
import json
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ===== GLOBALNE POSTAVKE =====
MAX_NEW_TOKENS = 1024
DEVICE_MAP = "auto"
MODEL_SAVE_PATH = r"E:\lillich\models\DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_DIR = "./outputs"
INPUT_FILE = "input.json"

# Postavke za suzbijanje upozorenja
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)

# Kreiraj direktorije ako ne postoje
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== UČITAVANJE ZADATAKA =====
def load_tasks(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        if not isinstance(tasks, list):
            raise ValueError("Input JSON should contain an array of tasks")
        
        for task in tasks:
            if 'task_id' not in task or 'messages' not in task:
                raise ValueError("Each task must have 'task_id' and 'messages' fields")
        
        return tasks
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {input_file}")

# ===== INICIJALIZACIJA MODELA =====
def init_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        trust_remote_code=True,
        cache_dir=MODEL_SAVE_PATH
    )

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        trust_remote_code=True,
        device_map=DEVICE_MAP,
        quantization_config=quant_config,
        cache_dir=MODEL_SAVE_PATH
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=DEVICE_MAP
    )

# ===== OBRAĐIVANJE ZADATAKA =====
def process_tasks(pipe, tasks):
    task_times = []
    
    for task in tasks:
        task_start = time.time()
        try:
            print(f"\nPočinjem zadatak {task['task_id']}...")
            
            output = pipe(
                task["messages"],
                max_new_tokens=MAX_NEW_TOKENS
            )
            
            assistant_reply = output[0]['generated_text'][-1]['content']
            task["messages"].append({"role": "assistant", "content": assistant_reply})
            
            save_results(task)
            
        except Exception as e:
            print(f"Greška u zadatku {task['task_id']}: {str(e)}")
        finally:
            task_time = time.time() - task_start
            task_times.append((task['task_id'], task_time))
            print(f"Zadatak {task['task_id']} završen u {task_time:.2f} sekundi")
    
    return task_times

# ===== SPREMANJE REZULTATA =====
def save_results(task):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/{timestamp}_task_{task['task_id']}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for msg in task["messages"]:
            f.write(f"== {msg['role'].upper()} ==\n")
            f.write(f"{msg['content']}\n\n")
        
        f.write("\n=== JSON FORMAT ===\n")
        json.dump(task["messages"], f, ensure_ascii=False, indent=2)

# ===== GLAVNI PROGRAM =====
if __name__ == "__main__":
    total_start = time.time()
    
    try:
        tasks = load_tasks(INPUT_FILE)
        print(f"Učitano {len(tasks)} zadataka iz {INPUT_FILE}")
        
        pipe = init_model()
        
        task_times = process_tasks(pipe, tasks)
        
        print("\n===== STATISTIKA IZVRŠAVANJA =====")
        total_time = time.time() - total_start
        for task_id, task_time in task_times:
            print(f"Zadatak {task_id}: {task_time:.2f}s ({task_time/total_time*100:.1f}%)")
        
        print(f"\nUKUPNO VRIJEME: {total_time:.2f} sekundi")
        print(f"Prosječno vrijeme po zadatku: {total_time/len(task_times):.2f}s")
        
    except Exception as e:
        print(f"Početna greška: {str(e)}")
    finally:
        torch.cuda.empty_cache()