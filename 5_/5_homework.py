import warnings
import os, transformers
os.environ["WANDB_DISABLED"] = "true"
transformers.logging.set_verbosity_error()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils")
import torch
import time
from datasets import load_dataset,Dataset,DatasetDict
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel)
from trl import SFTTrainer
#数据处理
raw_dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
print(raw_dataset)
train_data = raw_dataset["train"]
print("Instruction:",train_data[0]["instruction"])
print("Input:",train_data[0]["input"])
print("Output:",train_data[0]["output"])
print("Prompt:",train_data[0]["prompt"])
split_data=raw_dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_data["train"]
test_dataset = split_data["test"]
new_dataset_dict=DatasetDict({"train":train_dataset,"test":test_dataset})
print("new_dataset_dict",new_dataset_dict)
#抽取数据用于调试
debug_train_data_1k=new_dataset_dict["train"].select(range(500))
debug_test_data_1h=new_dataset_dict["test"].select(range(50))
debug_dataset_dict_1k=DatasetDict({
    "train":debug_train_data_1k,
    "test":debug_test_data_1h
})
print("debug_dataset_dict_1k",debug_dataset_dict_1k)
#Q处理
use_flash_attention=False
model_name="TinyPixel/Llama-2-7B-bf16-sharded"
bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_cpu_offload=False
)
#模型加载
model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True
)
model.config.pretraining_tp=1
print("model loaded")
model.gradient_checkpointing_enable()
model.config.use_cache = False
#分词器加载
tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"
print("tokenizer loaded")
#提示词格式
def format_instruction(sample):
    return f"""###INSTRUCTION:
You are an AI coding assistant specialized in generating python code from user instructions.Your task is to return only the code that directly fulfills the given instruction.</s>
###Input:
{sample['instruction']}</s>
###RESPONSE:
{sample['output']}</s>
"""
def generate_prompt(user_input):
    return f"""### INSTRUCTION:
You are an AI coding assistant specialized in generating python code from user instructions.Your task is to return only the code that directly fulfills the given instruction.</s>
###Input:
{user_input}</s>
###RESPONSE:
"""
#原始模型评估
logging.set_verbosity(logging.CRITICAL)
model.config.use_cache = True
prompt=generate_prompt(debug_dataset_dict_1k["test"][0]["instruction"])
pipe=pipeline(task="text-generation",
    tokenizer=tokenizer,
    model=model,
    max_length=512 )
result=pipe(prompt)
print(result[0]['generated_text'])
#清理数据
def clear_cache_and_collect():
    try:
        del model
    except NameError:
        pass
    try:
        del tokenizer
    except NameError:
        pass
    try:
        del trainer
    except NameError:
        pass
    while True:
        gc_collect=gc.collect()
        torch.cuda.empty_cache()
        if gc_collect == 0:
            break
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=0,abbreviated=True))
    print("Cache clearing,garbage collected,and variable deletion are complete.")
#训练时间
def timed_training(trainer):
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"花费{hours}:{minutes}:{seconds}")
#重新量化加载
torch.cuda.empty_cache()
gc.collect()
time.sleep(2)
model_name="TinyPixel/Llama-2-7B-bf16-sharded"
from accelerate import init_empty_weights
from accelerate.utils import get_balanced_memory

with init_empty_weights():
    dummy = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# ① 只给 gpu-0 分配显存
max_mem = {0: "8GiB"}
balanced_mem = get_balanced_memory(
    dummy,
    max_memory=max_mem,
    no_split_module_classes=["LlamaDecoderLayer"]
)

# ② 手工生成“全部在 0”的 map
device_map = {name: 0 for name, _ in dummy.named_modules() if len(list(name)) >= 0}
del dummy

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_cpu_offload=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    use_safetensors=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.config.pretraining_tp = 1

print("model loaded")
tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"
print("tokenizer loaded")
#LoRA配置
config=LoraConfig(
    r=2,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
#模型预处理
model=prepare_model_for_kbit_training(model)
model=get_peft_model(model,config)
#训练参数配置
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",      # ✅ 旧版关键字
    load_best_model_at_end=True,
    logging_strategy="steps",
    logging_steps=20,
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=0.3,
    lr_scheduler_type="constant",
    seed=42,
    save_total_limit=1,
    optim="paged_adamw_8bit"
)

#训练保存微调参数
trainer=SFTTrainer(
    model=model,
    train_dataset=debug_dataset_dict_1k["train"],
    eval_dataset=debug_dataset_dict_1k["test"],
    peft_config=config,
    formatting_func=format_instruction,
    tokenizer=tokenizer,
    max_seq_length=256,
    packing=True,
    args=training_arguments
)
model.config.use_cache = False
timed_training(trainer)
new_model_name="early_stopping_3_epoch_fine_tuned_llama"
trainer.model.save_pretrained(new_model_name)
clear_cache_and_collect()
#加载底座模型
device_map="auto"
base_model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map
)
#合并
model=PeftModel.from_pretrained(base_model,new_model_name)
model=model.merge_and_unload()
#微调后模型评估
prompt=generate_prompt(debug_dataset_dict_1k["test"][0]["instruction"])
pipe = pipeline(
    task="text-generation",
    model=model,
    max_length=256,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    device_map="auto"
)
result=pipe(prompt)
print(result[0]["generated_text"])







