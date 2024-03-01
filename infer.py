import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def do_inference(prompt):
        # Check for GPU availability and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        "LLM360/CrystalCoder",
        revision="CrystalCoder_phase1_checkpoint_055500",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "LLM360/CrystalCoder",
        revision="CrystalCoder_phase1_checkpoint_055500",
        trust_remote_code=True
    )
    model = model.to(device)


    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    gen_tokens = model.generate(input_ids, do_sample=True, max_length=400, pad_token_id=tokenizer.eos_token_id)

    return(tokenizer.batch_decode(gen_tokens)[0])
