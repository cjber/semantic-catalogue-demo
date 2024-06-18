import json
import logging
import os
from pathlib import Path

import dateparser
import polars as pl
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec

from src.common.utils import Consts, Paths, Settings


def _add_cdrc_metadata(doc_id: str) -> dict[str, str]:
    with open(Paths.DATA_DIR / "cdrc" / "catalogue-metadata.json") as f:
        catalogue_metadata = json.load(f)
    with open(Paths.DATA_DIR / "cdrc" / "files-metadata.json") as f:
        files_metadata = json.load(f)

    format, main_id = doc_id.split("-", maxsplit=1)

    if format != "notes":
        for file_meta in files_metadata:
            if main_id == file_meta["id"]:
                main_id = file_meta["parent_id"]
                break

    iso_date = dateparser.parse(files_metadata[0]["created"]).isoformat()
    for cm in catalogue_metadata:
        if main_id == cm["id"]:
            return {
                "title": cm["title"],
                "id": cm["id"],
                "url": cm["url"],
                "date_created": iso_date,
                "source": "CDRC",
            }
    raise ValueError(f"Metadata not found for document {doc_id}")


def _add_adr_metadata(filename: str) -> dict[str, str]:
    doc_id, origin_id, _ = Path(filename).stem.split("-")
    metadata = (
        pl.scan_parquet("./data/adr/adr_datasets.parquet")
        .filter((pl.col("id") == doc_id) & (pl.col("origin_id") == origin_id))
        .collect()
        .to_dict(as_series=False)
    )
    if len(metadata["id"]) == 0:
        return {
            "title": "",
            "id": f"{doc_id}-{origin_id}",
            "url": "",
            "date_created": "",
            "source": "ADR",
        }
    date_created = metadata["coverage"][0]["temporal"]["distributionReleaseDate"]
    date_created = (
        dateparser.parse(date_created).isoformat()
        if isinstance(date_created, str)
        else ""
    )
    return {
        "title": metadata["name"][0],
        "id": f"{doc_id}-{origin_id}",
        "url": metadata["url"][0],
        "date_created": date_created,
        "source": "ADR",
    }


class CreateDataStore:
    def __init__(
        self,
        index_name: str,
        chunk_size: int,
        chunk_overlap: int,
        overwrite: bool,
        embed_dim: int,
        cdrc_profiles_dir: Path = Paths.CDRC_PROFILES_DIR,
        adr_profiles_dir: Path = Paths.ADR_DIR,
        pipeline_storage: Path = Paths.PIPELINE_STORAGE,
    ):
        self.index_name = index_name
        self.overwrite = overwrite
        self.cdrc_profiles_dir = cdrc_profiles_dir
        self.adr_profiles_dir = adr_profiles_dir
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pipeline_storage = pipeline_storage

        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    def run(self):
        if not self.cdrc_profiles_dir.exists() or not self.adr_profiles_dir.exists():
            return logging.error("Data directories do not exist.")
        self.initialise_pinecone_index()
        self.setup_directory_reader()
        self.setup_ingestion_pipeline()
        self.load_and_preprocess_documents()

    def initialise_pinecone_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed_dim,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                metric="cosine",
            )
        elif self.overwrite:
            self.pc.delete_index(self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed_dim,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                metric="cosine",
            )

    def setup_directory_reader(self):
        # pdf_reader = LlamaParse()
        pdf_reader = UnstructuredReader()
        self.cdrc_dir_reader = SimpleDirectoryReader(
            str(self.cdrc_profiles_dir),
            recursive=True,
            file_extractor={".pdf": pdf_reader},
            file_metadata=lambda name: _add_cdrc_metadata(Path(name).stem),
        )
        self.adr_dir_reader = SimpleDirectoryReader(
            str(self.adr_profiles_dir),
            recursive=True,
            file_metadata=lambda name: _add_adr_metadata(name),
        )

    def setup_ingestion_pipeline(self):
        self.vector_store = PineconeVectorStore(
            pinecone_index=self.pc.Index(self.index_name)
        )
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
                OpenAIEmbedding(
                    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
                    model=Consts.OPENAI_EMBEDDING_MODEL,
                    api_key=os.environ["OPENAI_API_KEY"],
                ),
            ],
            vector_store=self.vector_store,
        )

    def load_and_preprocess_documents(self):
        self.docs = self.cdrc_dir_reader.load_data(show_progress=True)
        self.docs.extend(self.adr_dir_reader.load_data(show_progress=True))
        for doc in self.docs:
            doc.excluded_embed_metadata_keys.extend(
                ["id", "url", "filename", "date_created"]
            )
            doc.excluded_llm_metadata_keys.extend(
                ["id", "url", "filename", "date_created"]
            )

        self.pipeline.run(documents=self.docs)


if __name__ == "__main__":
    datastore = CreateDataStore(**Settings().datastore.model_dump())
    datastore.run()
