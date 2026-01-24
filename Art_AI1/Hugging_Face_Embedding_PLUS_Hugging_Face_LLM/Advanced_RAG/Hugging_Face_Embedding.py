from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

pd.set_option(
    "display.max_colwidth", None
)  # This will be helpful when visualizing retriever outputs

pdf_path = Path("Sigmund.pdf")
assert pdf_path.exists(), f"PDF not found: {pdf_path.resolve()}"
RAW_KNOWLEDGE_BASE = PyPDFLoader(str(pdf_path)).load()
#print(RAW_KNOWLEDGE_BASE)
################################################################
### Step 1
### Here will base on the text that imported do the chunking
print("Step 1")
print("#################################################################")
from langchain.text_splitter import RecursiveCharacterTextSplitter
# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
    chunk_overlap=100,  # The number of characters to overlap between chunks
    add_start_index=True,  # If `True`, includes chunk's start index in metadata
    strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    separators=MARKDOWN_SEPARATORS,
)

docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])
