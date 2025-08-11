import sys

print("=== PYTHON ENVIRONMENT INFO ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

# Try to import and print package versions + file paths
def check_package(pkg_name):
    try:
        pkg = __import__(pkg_name)
        print(f"{pkg_name} version:", getattr(pkg, "__version__", "unknown"))
        print(f"{pkg_name} location:", getattr(pkg, "__file__", "built-in"))
    except ImportError as e:
        print(f"{pkg_name} NOT INSTALLED:", e)
    print()

print("=== PACKAGE VERSIONS ===")
for name in ["sentence_transformers", "transformers", "huggingface_hub", "langchain"]:
    check_package(name)

# Quick embedding test
print("=== TEST EMBEDDING CREATION ===")
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    test_vec = emb.embed_query("Hello world")
    print("Embedding length:", len(test_vec))
    print("First 5 dims:", test_vec[:5])
    print("✅ Embedding model loaded successfully.")
except Exception as e:
    print("❌ Failed to load embedding model:", e)
