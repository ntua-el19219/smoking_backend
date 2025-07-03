from chromaClient import get_chroma_collection

if __name__ == "__main__":
    kb = get_chroma_collection("knowledge_base")

    # H get() επιστρέφει όλα τα documents
    docs = kb.get()
    count = len(docs["documents"])
    print(f"📦 Total documents in collection: {count}")

    # Δείγμα retrieval
    if count > 0:
        query = "What are some health benefits of quitting smoking?"
        results = kb.similarity_search(query, k=3)

        print("\n🔍 Top results:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content[:400])
            print(f"Source: {doc.metadata.get('source')}")
