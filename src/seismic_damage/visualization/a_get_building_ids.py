from seismic_damage.pipelines.rag.pipeline import RAGPipeline

if __name__ == '__main__':
    rag = RAGPipeline()
    # Ensure the vector store is indexed; if not, index now using default chunks dir
    if rag.store is None:
        rag.index_corpus()  # this will load default chunks or knowledge dir and build FAISS
    result = rag.run('List every unique building identifier mentioned in these Bhuj 2001 reports.')
    print(result.answer)
