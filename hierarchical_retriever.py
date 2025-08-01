import ast

class SingleRetriever():
    def __init__(self, vector_store, top_k, level_name, filter_indices = None):
        self.vector_store = vector_store
        self.top_k = top_k
        self.level_name = level_name
        self.filter_indices = filter_indices
    
    def get_indices(self):
        return self.filter_indices
    
    def update_indices(self, indices):
        if isinstance(indices, str):
            indices = ast.literal_eval(indices)
        if isinstance(indices, list):
            self.filter_indices = indices
        else:
            raise ValueError("filter_indices must be a list.")
        
    def remove_indices(self):
        self.filter_indices = None

    def retrieve_documents(self, query):
        if self.filter_indices:
            response = self.vector_store.similarity_search(
                query=query,
                k=self.top_k,
                filter={"chunk_index": {"$in": self.filter_indices}}
            )
        else:
            response = self.vector_store.similarity_search(
                query=query,
                k=self.top_k
            )
        return response

class HierarchicalRetriever:
    def __init__(self, single_retrievers):
        self.retrievers_map = {
            retriever.level_name: retriever for retriever in single_retrievers
        }

    def get_retriever(self, level):
        if level not in self.retrievers_map:
            raise KeyError(f"{level} not found in retrievers.")
        return self.retrievers_map[level]

    def update_indices(self, level, indices):
        self.get_retriever(level).update_indices(indices)


    def remove_indices(self, level):
        self.get_retriever(level).remove_indices()

    def get_indices(self, level):
        return self.get_retriever(level).get_indices()

    def retrieve_by_level(self, query, level, indices = None):
        if indices and len(indices) > 0:
            self.update_indices(level, indices=indices)
        
        results = self.get_retriever(level).retrieve_documents(query)
        self.remove_indices(level)
        return results