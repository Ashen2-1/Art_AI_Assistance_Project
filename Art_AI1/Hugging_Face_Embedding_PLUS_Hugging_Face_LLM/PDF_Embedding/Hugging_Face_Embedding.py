################################################################
### Step 1
### Here we will load in the sources (Here we load is a PDF) and we will chunk it for later Hugging Face model to embed
print("Step 1")
print("#################################################################")
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
### We will first test using a PDF
### PyPDFLoader from LangChain to extract the text from the PDF
path = "Sigmund.pdf"
#path = "barthes.pdf"
try:
    loader = PyPDFLoader(path)
    docs = loader.load()
    if not docs or not docs[0].page_content.strip():
        raise ValueError("Empty text, fallback to OCR")
except Exception:
    loader = UnstructuredPDFLoader(path, strategy="ocr_only")
    docs = loader.load()

# loader = PyPDFLoader(path)
# docs = loader.load()
# print(len(docs))

# def extract_text_from_file(path: str, mime_type: str) -> str:
#     if mime_type == "application/pdf":
#         return extract_text_from_pdf(path)
#     elif mime_type.startswith("image/"):
#         return extract_text_from_image(path)
#     elif mime_type in ("text/plain", "text/markdown"):
#         return open(path, "r", encoding="utf-8").read()
#     elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#         return extract_text_from_docx(path)
#     else:
#         raise ValueError(f"Unsupported file type: {mime_type}")


from langchain_text_splitters import RecursiveCharacterTextSplitter
### Then split the text into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
### We set the chunk size as 1000 and the overlap as 200
### which means each chunk will nearly have 1000 characters and the overlap between two chunks will be 200 characters.

chunks = text_splitter.split_documents(docs)
### Here we will start chunking the doc
print(len(chunks))
text_lines = [chunk.page_content for chunk in chunks]
print(len(text_lines))
### Here we put it as a list of text so Hugging Face embedding model will easily embed
################################################################
################################################################
### Step 2
### Here we will get up the embedding model
print("Step 2")
print("#################################################################")
from sentence_transformers import SentenceTransformer
### We use BGE embedding model as an example
### But we can use any embedding models, such as those found on the MTEB leaderboard (https://huggingface.co/spaces/mteb/leaderboard).
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]
### This function will embed the text e.g."[-0.07660683244466782, 0.025316666811704636, 0.012505513615906239, 0.004595153499394655, 0.025780051946640015, 0.03816710412502289, 0.08050819486379623, 0.003035430097952485, 0.02439221926033497, 0.0048803347162902355]"
test_embedding = emb_text("This is a test") ### Here we can change to User input
embedding_dim = len(test_embedding)
################################################################
################################################################
### Step 3
### Here we will start to set up the Milvus RAG vector database
print("Step 3")
print("#################################################################")
### Here we will use the chromadb vector database for RAG
## from pymilvus import MilvusClient
import chromadb
chroma_client = chromadb.Client()

collection_name = "rag_collection"

# 如果你想每次重建，可以先删一下（chroma 是 reset 全部）
# chroma_client.reset()

# 获取或创建一个 collection
collection = chroma_client.get_or_create_collection(
    name=collection_name
)
### Here we will build a new vector database named "rag_collection"
################################################################
################################################################
### Step 4
### Here we will embed the PDF that we imported earlier and have a loading bar
print("Step 4")
print("#################################################################")
from tqdm import tqdm

ids = []
embeddings = []
metadatas = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    ids.append(str(i))                # Chroma 的 id 要是 str
    embeddings.append(emb_text(line)) # 我们已经自己算好 embedding
    metadatas.append({"text": line})  # 把原始文本放进 metadata
    print("Chunk: ", i,"=============================")
    print(line)

# 一次性插入
collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
)
print(f"Inserted {len(ids)} chunks into Chroma.")
################################################################
################################################################
### Step 5
### Here we will set up the question and base on the question find the most related content for future answer
print("Step 5")
print("#################################################################")
question = "Can you explain uncanny"
### Here is base on the users text

### Here we will search for the question in the collection and retrieve the top 3 semantic matches.
query_embedding = emb_text(question)

search_res = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["metadatas", "distances"],  # 我们需要文本 + 距离
)
################################################################
################################################################
### Step 6
###
print("Step 6")
print("#################################################################")
import json
metadatas = search_res["metadatas"][0]
distances = search_res["distances"][0]

retrieved_lines_with_distances = [
    (meta["text"], dist) for meta, dist in zip(metadatas, distances)
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
### Here we can print out the three most related searches

context = "\n".join([t for (t, _) in retrieved_lines_with_distances])
### Here we will only bring out the context part and join it e.x("chunk text 1 ...\nchunk text 2 ...\nchunk text 3 ...")
PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

user_prompt = PROMPT.format(context=context, question=question)
print(user_prompt)
import transformers
import torch
import os

model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=0,
    dtype=torch.bfloat16,
)
system_msg = (
    "You are an AI assistant for art history research. "
    "You must answer ONLY based on the provided context. "
    "You must ONLY answer the question that user asked."
    "If the answer is not in the context, say you don't know."
    "Do NOT repeat the prompt, tags, or the word 'context'. "
    "Write 2–4 short paragraphs and end with a single-sentence conclusion."
)
full_prompt = f"{system_msg}\n\n{user_prompt}\n\nAnswer:"
outputs = pipe(
    full_prompt,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.2,
    return_full_text=False,

)
answer = outputs[0]["generated_text"].strip()
print("\n===== YOUR QUESTION =====\n")
print(question)
print("\n===== MODEL ANSWER =====\n")
import textwrap

print(textwrap.fill(answer, width=80))