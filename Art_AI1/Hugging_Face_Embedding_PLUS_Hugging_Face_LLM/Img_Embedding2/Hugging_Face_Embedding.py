import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import LlavaForConditionalGeneration

def answer_with_vlm_only(image_path: str, question: str) -> str:
    model_id = "llava-hf/llava-1.5-7b-hf"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Reinstall a CUDA-enabled PyTorch first.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        low_cpu_mem_usage=True,
        dtype=torch.float16,     # <- 用 dtype
    ).to("cuda")                # <- 明确放到GPU

    image = Image.open(image_path).convert("RGB")

    system_msg = (
        "You are an AI assistant for art history research.\n"
        "You must answer ONLY based on the user-provided image.\n"
        "If something is unclear, say what is unclear instead of inventing.\n"

    )
    prompt = f"USER: <image>\n{system_msg}\nQuestion: {question}\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7,top_p=0.9,repetition_penalty=1.05,)

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


import textwrap
print(textwrap.fill(answer_with_vlm_only("Artwork.webp", "Can you talk about this artwork and give me some other examples and talk about it.")))
