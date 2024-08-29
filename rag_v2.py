import json
import logging
import re
from typing import List, Dict, Any

# Import necessary components from llama_index
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import KeywordExtractor
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llm_utils.model_loaders import ModelInitializer

# Set up logging to display detailed information
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomHybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        weight_vector: float = 0.5,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weight_vector = weight_vector
        logger.info(
            f"Initialized CustomHybridRetriever with vector weight: {weight_vector}"
        )

    def _retrieve(self, query: QueryBundle, **kwargs) -> List[NodeWithScore]:
        logger.info(f"Retrieving nodes for query: {query.query_str}")
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        bm25_nodes = self.bm25_retriever.retrieve(query.query_str, **kwargs)

        # Combine and deduplicate results
        node_dict = {}
        for node in vector_nodes + bm25_nodes:
            if node.node.node_id not in node_dict:
                node_dict[node.node.node_id] = node
            else:
                existing_node = node_dict[node.node.node_id]
                vector_score = (
                    node.score if node in vector_nodes else existing_node.score
                )
                bm25_score = node.score if node in bm25_nodes else existing_node.score

                # Normalize scores
                max_score = max(vector_score, bm25_score)
                vector_score_normalized = (
                    vector_score / max_score if max_score != 0 else 0
                )
                bm25_score_normalized = bm25_score / max_score if max_score != 0 else 0

                combined_score = (
                    self.weight_vector * vector_score_normalized
                    + (1 - self.weight_vector) * bm25_score_normalized
                )
                node_dict[node.node.node_id] = NodeWithScore(
                    node=existing_node.node, score=combined_score
                )

        # Sort by score and return
        return sorted(node_dict.values(), key=lambda x: x.score, reverse=True)


class RAGApplication:
    def __init__(self, json_file_path: str, vector_store_dir: str):
        self.json_file_path = json_file_path
        self.vector_store_dir = vector_store_dir
        # trial
        self.json_data = []
        self.documents = []
        self.nodes = []
        self.vector_index = None
        self.hybrid_retriever = None
        self.query_engine = None

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = self.embed_model

    def load_json_data(self) -> List[Dict[str, Any]]:
        logger.info(f"Loading JSON data from {self.json_file_path}")
        with open(self.json_file_path, "r") as f:
            json_data = json.load(f)
        self.json_data = json_data
        logger.info(f"Loaded {len(json_data)} records from JSON")
        return self.json_data

    def clean_text(self, text: str) -> str:
        return re.sub(r"[\n\t]+", " ", text.strip())

    def create_documents(
        self, json_data: List[Dict[str, Any]], embed_field: str = "article_text"
    ) -> List[Document]:
        logger.info("Creating documents from JSON data...")
        self.documents = [
            Document(
                text=self.clean_text(item[embed_field]),
                metadata={
                    "article_title": self.clean_text(item["article_title"]),
                    "article_number": item["article_number"],
                    "article_url": item["article_url"],
                    "article_citation": self.clean_text(item["article_citation"]),
                },
            )
            for item in json_data
        ]
        # Log details of documents created
        for i, doc in enumerate(self.documents[:5]):  # Print first 5 for brevity
            logger.info(f"Document {i}: {doc}")
        return self.documents

    async def create_nodes(self, chunk_size: int = 512, chunk_overlap: int = 20):
        logger.info("Creating nodes from documents...")
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        keyword_extractor = KeywordExtractor(
            keywords=5,
            llm=ModelInitializer.initialize_groq(
                from_llamaindex=True,
                model="llama-3.1-8b-instant",
            ),
        )

        ingestion_pipeline = IngestionPipeline(
            transformations=[node_parser, keyword_extractor]
        )

        self.nodes = await ingestion_pipeline.arun(documents=self.documents)
        logger.info(f"Created {len(self.nodes)} nodes")

        # Log details of nodes created
        for i, node in enumerate(self.nodes[:5]):  # Print first 5 for brevity
            logger.info(
                f"Node {i} (Before Embedding): {node}\nMetadata: {node.get_metadata_str()}"
            )
        return self.nodes

    def create_or_load_vector_index(self, index_id: str = "vector_index"):
        logger.info(f"Creating or loading vector index from {self.vector_store_dir}")
        # storage_context = StorageContext.from_defaults(
        #     persist_dir=self.vector_store_dir
        # )

        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.vector_store_dir
            )
            self.vector_index = load_index_from_storage(
                storage_context, index_id=index_id
            )
            logger.info("Loaded existing index from storage")
        except Exception as e:
            logger.warning(
                f"Failed to load index from storage: {e}. Creating new index..."
            )
            # storage_context = StorageContext.from_defaults(
            #     persist_dir=self.vector_store_dir
            # )
            self.vector_index = VectorStoreIndex(
                nodes=self.nodes,
                # storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True,
            )
            self.vector_index.set_index_id(index_id)
            self.vector_index.storage_context.persist(persist_dir=self.vector_store_dir)
            logger.info(f"Created and persisted new index to {self.vector_store_dir}")

        # Log vector index details
        logger.info(f"Vector Index: {self.vector_index}")
        return self.vector_index

    def create_hybrid_retriever(
        self, similarity_top_k: int = 5, weight_vector: float = 0.5
    ):
        logger.info("Creating hybrid retriever")
        vector_retriever = VectorIndexRetriever(
            index=self.vector_index, similarity_top_k=similarity_top_k
        )
        bm25_retriever = BM25Retriever.from_defaults(
            index=self.vector_index, similarity_top_k=similarity_top_k
        )
        self.hybrid_retriever = CustomHybridRetriever(
            vector_retriever, bm25_retriever, weight_vector=weight_vector
        )

        # Log retrievers details
        logger.info(f"Vector Retriever: {vector_retriever}")
        logger.info(f"BM25 Retriever: {bm25_retriever}")
        return self.hybrid_retriever

    def create_query_engine(self, use_rerank: bool = False):
        logger.info("Creating query engine")
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.7)]

        if use_rerank:
            colbert_reranker = ColbertRerank(top_n=5, model="colbert-ir/colbertv2.0")
            postprocessors.append(colbert_reranker)

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.hybrid_retriever,
            node_postprocessors=postprocessors,
            llm=ModelInitializer.initialize_groq(
                from_llamaindex=True,
                model="llama-3.1-70b-versatile",
            ),
        )

        # Log query engine details
        logger.info(f"Query Engine: {self.query_engine}")
        return self.query_engine

    def retrieve_and_display(self, query: str, num_results: int = 5):
        logger.info(f"Retrieving results for query: {query}")
        retrieved_nodes = self.query_engine.retrieve(query)

        print(f"Top {num_results} results for query: '{query}'\n")
        for i, node in enumerate(retrieved_nodes[:num_results], 1):
            print(f"Result {i}:")
            print(node)
            print(node.metadata)
            # print(f"Score: {node.score}")
            # print(f"Content: {node.node.text[:200]}...")
            # print(f"Metadata: {node.node.metadata}")
            print("-" * 50)

        # Log retrieved nodes
        for i, node in enumerate(retrieved_nodes[:num_results]):
            logger.info(f"Retrieved Node {i}: {node}")


def main():
    json_file_path = "data//scraped_data//test_data_final_myasthenia.json"
    rag_app = RAGApplication(
        json_file_path, vector_store_dir="src\\rag_app\\vector_stores"
    )

    # Load JSON data
    json_data = rag_app.load_json_data()

    # Create documents (customize the embed_field if needed)
    rag_app.create_documents(json_data, embed_field="article_text")

    # Create nodes
    rag_app.create_nodes(chunk_size=512, chunk_overlap=20)

    # Create or load vector index
    rag_app.create_or_load_vector_index()

    # Create hybrid retriever
    rag_app.create_hybrid_retriever(similarity_top_k=5, weight_vector=0.5)

    # Create query engine (set use_rerank to True to enable reranking)
    rag_app.create_query_engine(use_rerank=False)

    # Retrieve and display results
    rag_app.retrieve_and_display(
        "what's the prevalence of myasthenia gravis in USA", num_results=5
    )


if __name__ == "__main__":
    main()
