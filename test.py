import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from peft import PeftModel, PeftConfig


# finetune_model_path = "/mnt/f/TinyLlama/models/TinyLlama-1.1B-Chat-v1.0"  # For example: 'FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
# # config = PeftConfig.from_pretrained(finetune_model_path)
# tokenizer = AutoTokenizer.from_pretrained(finetune_model_path, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(finetune_model_path, device_map='auto', torch_dtype=torch.float16, load_in_8bit=False)
# # model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
# # model = model.eval()

# input_ids = tokenizer(['What is Techcombank in VietNam'], return_tensors="pt", add_special_tokens=False).input_ids.to('cuda') 

# print('------ TEST -----------')
# print(input_ids)

# generate_input = {
#     "input_ids": input_ids,
#     "max_new_tokens": 512,
#     "do_sample": True,
#     "top_k": 50,
#     "top_p": 0.95,
#     "temperature": 0.3,
#     "repetition_penalty": 1.3,
#     "eos_token_id": tokenizer.eos_token_id,
#     "bos_token_id": tokenizer.bos_token_id,
#     "pad_token_id": tokenizer.pad_token_id
# }
# generate_ids = model.generate(**generate_input)
# text = tokenizer.decode(generate_ids[0])
# print(text)

# model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map='auto', torch_dtype=torch.float16, load_in_8bit=False)



# # # tokenizer=AutoTokenizer.from_pretrained("/mnt/f/TinyLlama/models/TinyLlama-1.1B-Chat-v1.0")

# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "What is TCB in VietNam?"},
# ]

# # input_ids = tokenizer.encode(messages, truncation=True, max_length=500)

# # params = {
# #     "prompt": prompt_text,
# #     "n_predict": n_predict,
# #     "n_ctx": context_length
# # }

# pipe = pipeline("text-generation", model='./models/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float, device_map="auto")


# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# # input_ids = pipe.tokenizer(['<s>Human: What is Techcombank?\n</s><s>Assistant: '], return_tensors="pt", add_special_tokens=False).input_ids.to('mps')

# print('------ TEST -----------')
# print(prompt)


# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])
# # # <|system|>
# # # You are a friendly chatbot who always responds in the style of a pirate.</s>
# # # <|user|>
# # # How many helicopters can a human eat in one sitting?</s>
# # # <|assistant|>
# # # ...


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained('/mnt/f/TinyLlama/models/TinyLlama-1.1B-Chat-v1.0', device_map='auto', torch_dtype=torch.float16, load_in_8bit=False)
# model = model.eval()
# tokenizer = AutoTokenizer.from_pretrained('/mnt/f/TinyLlama/models/TinyLlama-1.1B-Chat-v1.0', use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token

# messages = [
#     # {
#     #     "role": "system",
#     #     "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     # },
#     {"role": "user", "content": "What is Techcombank?"},
# ]

# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(prompt)


# input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')


# generate_input = {
#     "input_ids": input_ids,
#     "max_new_tokens": 512,
#     "do_sample": True,
#     "top_k": 50,
#     "top_p": 0.95,
#     "temperature": 0.3,
#     "repetition_penalty": 1.3,
#     "eos_token_id": tokenizer.eos_token_id,
#     "bos_token_id": tokenizer.bos_token_id,
#     "pad_token_id": tokenizer.pad_token_id
# }

# generate_ids  = model.generate(**generate_input)
# text = tokenizer.decode(generate_ids[0])
# print(text)











# tests = [['Hi', 'How to use the "if" statement in Python?\nThe if-else statement is a powerful construct that allows you to execute code based on certain conditions. In this tutorial, we\'ll learn how to write an example using the `if` and `elif` statements with some examples of their usage. Let us begin!\nIn our program, let’s say we want to check whether the user has entered any valid input or not. We can do it by checking for empty strings (which are always false). If there’s no string value at all, then print “Invalid Input” message along with exiting from the function.\nLet me know your thoughts about my explanation above. Please share more such tutorials related to programming languages like Java, C++, PHP etc., so I could add them here as well. Thank You !!!</s>'], ['a', None]]

# messages = [
#             # {
#             #     "role": "system",
#             #     "content": "You are a friendly chatbot who always responds in the style of a pirate",
#             # },
#             # {"role": "user", "content": "What is Techcombank?"},
#         ]

# for i, n in enumerate(tests):
#     for j, m in enumerate(n):
#         if m is not None:
#             messages.append({"role": "user" if j % 2 == 0 else "assistant", "content": m.replace('<s>','').replace('</s>','')})


# # history_true = tests[-1][0].index

# # print(history_true)

# # messages = [{
# #                 "role": "User" if i % 2 == 0 else "Assistant",
# #                 "content": n,
# #             } for i,n in filter((x) => {}, enumerate(tests)) if n != None]



# print(messages)

# model_name_or_path = "./models/TinyLlama-1.1B-Chat-v1.0"

# # Load model
# # model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
# #                                           trust_remote_code=False, safetensors=True)

# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', torch_dtype=torch.float16, load_in_8bit=False)
# model = model.eval()

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

# prompt = "What is Techcombank?"
# prompt_template=f'''{prompt}

# '''

# print("\n\n*** Generate:")

# tokens = tokenizer(
#     prompt_template,
#     return_tensors='pt'
# ).input_ids.to('mps')

# # Generate output
# generation_output = model.generate(
#     tokens,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.95,
#     top_k=40,
#     max_new_tokens=512
# )

# print("Output: ", tokenizer.decode(generation_output[0]))



# from langchain_community.llms import VLLM

# llm = VLLM(
#     model="./models/TinyLlama-1.1B-Chat-v1.0",
#     trust_remote_code=True,  # mandatory for hf models
#     max_new_tokens=128,
#     top_k=10,
#     top_p=0.95,
#     temperature=0.8,
# )

# print(llm("What is the capital of France ?"))