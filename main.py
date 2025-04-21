import os
import pickle
import faiss
import numpy as np
import hashlib
from uuid import uuid4
import uvicorn
from typing import Dict, List, Optional, Any, Set
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.collection import Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class Config:
    """Configuration class to manage all environment variables and constants"""
    
    # File and model settings
    TXT_FOLDER = "data"
    EMBED_MODEL = "text-embedding-3-small"
    GPT_MODEL = "gpt-4o"
    INDEX_PATH = "faiss_index"
    DOCUMENT_REGISTRY_PATH = "document_registry.pkl"
    
    # API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # MongoDB settings
    MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
    DB_NAME = "property_finder_ai"
    CHAT_HISTORY_COLLECTION = "chat_histories"
    DOCUMENT_REGISTRY_COLLECTION = "document_registry"
    
    # Constants
    MAX_HISTORY_MESSAGES = 20
    RETRIEVAL_K = 3
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # System prompts
    SYSTEM_PROMPT = """You are Property Finder's AI assistant, the leading real estate platform in the UAE. 
    Your main role is to provide detailed information about different communities and neighborhoods across Dubai, 
    Sharjah, Abu Dhabi, Ajman, and Ras Al Khaimah. Always maintain a professional and 
    knowledgeable tone as a leading property portal in the UAE.
    Use context provided to answer the user's question.
    Do not answer any questions outside of the context provided.
    If you do not know the answer, say "I don't know."
    Use history of the conversation to provide a more personalized response."""
    
    STANDALONE_QUESTION_SYSTEM_PROMPT = """Given a chat history and the latest user question, 
    formulate a standalone question that includes all context needed to retrieve relevant information to answer the question."""


class DocumentRegistry:
    """Manages tracking of processed documents to avoid reprocessing"""
    
    def __init__(self, db_manager=None):
        """Initialize document registry"""
        self.db_manager = db_manager
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, str]:
        """Load registry from database or file"""
        if self.db_manager:
            # Use MongoDB
            registry_data = self.db_manager.get_document_registry()
            return registry_data if registry_data else {}
        else:
            # Use file-based registry
            if os.path.exists(Config.DOCUMENT_REGISTRY_PATH):
                with open(Config.DOCUMENT_REGISTRY_PATH, "rb") as f:
                    return pickle.load(f)
            return {}
    
    def _save_registry(self):
        """Save registry to database or file"""
        if self.db_manager:
            # Save to MongoDB
            self.db_manager.update_document_registry(self.registry)
        else:
            # Save to file
            with open(Config.DOCUMENT_REGISTRY_PATH, "wb") as f:
                pickle.dump(self.registry, f)
    
    def get_document_hash(self, file_path: str, content: str) -> str:
        """Create a unique hash for a document based on path and content"""
        return hashlib.md5(f"{file_path}:{content}".encode()).hexdigest()
    
    def is_document_processed(self, file_path: str, content: str) -> bool:
        """Check if a document has been processed before"""
        doc_hash = self.get_document_hash(file_path, content)
        return doc_hash in self.registry
    
    def register_document(self, file_path: str, content: str):
        """Mark a document as processed"""
        doc_hash = self.get_document_hash(file_path, content)
        self.registry[doc_hash] = {
            "path": file_path,
            "processed_at": datetime.utcnow().isoformat(),
            "size": len(content)
        }
        self._save_registry()
    
    def get_all_registered_paths(self) -> Set[str]:
        """Get all registered file paths"""
        return {info["path"] for info in self.registry.values()}


class DocumentLoader:
    """Handles loading and processing documents from files"""
    
    def __init__(self, registry: DocumentRegistry):
        """Initialize with document registry"""
        self.registry = registry
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
    def load_text_files(self, folder_path: str, process_all: bool = False) -> List[Document]:
        """Load all text files from a folder and convert them to Document objects"""
        documents = []
        
        # Load raw text from files
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Skip if already processed and we're not forcing reprocessing
                if not process_all and self.registry.is_document_processed(file_path, content):
                    continue
                
                # Process this document
                chunks = self.text_splitter.split_text(content)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": file_path}
                    ))
                
                # Register this document
                self.registry.register_document(file_path, content)
        
        return documents
    
    def process_new_document(self, file_path: str, content: str) -> List[Document]:
        """Process a single new document"""
        # Skip if already processed
        if self.registry.is_document_processed(file_path, content):
            return []
        
        # Process the document
        chunks = self.text_splitter.split_text(content)
        documents = [
            Document(page_content=chunk, metadata={"source": file_path})
            for chunk in chunks
        ]
        
        # Register the document
        self.registry.register_document(file_path, content)
        
        return documents


class VectorStoreManager:
    """Manages vector embeddings and retrieval operations"""
    
    def __init__(self, document_loader: DocumentLoader):
        """Initialize the vector store manager"""
        self.document_loader = document_loader
        self.embeddings = OpenAIEmbeddings(model=Config.EMBED_MODEL)
        self.vectorstore = self._load_or_create_vectorstore()
        # Cache of vectorstores by session ID
        self.session_vectorstores: Dict[str, FAISS] = {}
    
    def _load_or_create_vectorstore(self) -> FAISS:
        """Load existing vectorstore or create a new one"""
        if os.path.exists(Config.INDEX_PATH):
            print("✅ Loading stored vector index...")
            return FAISS.load_local(Config.INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("🔄 Building new vector index...")
            documents = self.document_loader.load_text_files(Config.TXT_FOLDER, process_all=True)
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.save_local(Config.INDEX_PATH)
            print(f"✅ Built and saved vector store with {len(documents)} chunks.")
            return vectorstore
    
    def get_vectorstore_for_session(self, session_id: str) -> FAISS:
        """Get a vectorstore for a specific session (using caching)"""
        if session_id not in self.session_vectorstores:
            self.session_vectorstores[session_id] = self.vectorstore
        return self.session_vectorstores[session_id]
    
    def add_documents_to_vectorstore(self, documents: List[Document]) -> int:
        """Add new documents to the existing vectorstore"""
        if not documents:
            return 0
        
        # Add to the main vectorstore
        self.vectorstore.add_documents(documents)
        
        # Save the updated vectorstore
        self.vectorstore.save_local(Config.INDEX_PATH)
        
        # Clear the session cache to ensure all sessions get the updated vectorstore
        self.session_vectorstores = {}
        
        return len(documents)
    
    def process_folder_for_new_documents(self, folder_path: str = Config.TXT_FOLDER) -> int:
        """Process a folder for new documents and add them to the vectorstore"""
        new_documents = self.document_loader.load_text_files(folder_path)
        return self.add_documents_to_vectorstore(new_documents)
    
    def process_new_document_content(self, file_path: str, content: str) -> int:
        """Process a single new document and add it to the vectorstore"""
        documents = self.document_loader.process_new_document(file_path, content)
        return self.add_documents_to_vectorstore(documents)


class DatabaseManager:
    """Handles all database operations for chat history and document registry"""
    
    def __init__(self):
        """Initialize database connection and collections"""
        self.client = MongoClient(Config.MONGODB_CONNECTION_STRING)
        self.db = self.client[Config.DB_NAME]
        self.chat_collection = self.db[Config.CHAT_HISTORY_COLLECTION]
        self.doc_registry_collection = self.db[Config.DOCUMENT_REGISTRY_COLLECTION]
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        self.chat_collection.create_index("session_id", unique=True)
        self.chat_collection.create_index("updated_at")
        self.doc_registry_collection.create_index("path", unique=True)
    
    def get_or_create_session(self, session_id: str) -> dict:
        """Get an existing session or create a new one"""
        session = self.chat_collection.find_one({"session_id": session_id})
        if not session:
            session = {
                "session_id": session_id,
                "messages": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            self.chat_collection.insert_one(session)
        return session
    
    def get_chat_history(self, session_id: str) -> List[dict]:
        """Get chat history for a session"""
        session = self.get_or_create_session(session_id)
        return session.get("messages", [])
    
    def add_message_pair(self, session_id: str, user_message: str, ai_message: str):
        """Add a pair of user and AI messages to history"""
        timestamp = datetime.utcnow()
        
        # Update the session with new messages
        self.chat_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_message, "timestamp": timestamp},
                            {"role": "assistant", "content": ai_message, "timestamp": timestamp}
                        ]
                    }
                },
                "$set": {"updated_at": timestamp}
            }
        )
        
        # Trim history if needed
        self._trim_history(session_id)
    
    def _trim_history(self, session_id: str):
        """Ensure history doesn't exceed maximum size"""
        session = self.chat_collection.find_one({"session_id": session_id})
        if session and len(session["messages"]) > Config.MAX_HISTORY_MESSAGES:
            # Keep only the most recent messages
            latest_messages = session["messages"][-Config.MAX_HISTORY_MESSAGES:]
            self.chat_collection.update_one(
                {"session_id": session_id},
                {"$set": {"messages": latest_messages}}
            )
    
    def create_new_session(self) -> str:
        """Create a new chat session and return its ID"""
        session_id = str(uuid4())
        self.get_or_create_session(session_id)
        return session_id
    
    def cleanup_old_sessions(self, days_threshold: int = 30):
        """Remove chat sessions older than the specified threshold"""
        threshold_date = datetime.utcnow() - timedelta(days=days_threshold)
        result = self.chat_collection.delete_many({"updated_at": {"$lt": threshold_date}})
        print(f"Removed {result.deleted_count} old chat sessions")
    
    def get_document_registry(self) -> Dict[str, Any]:
        """Get the document registry from the database"""
        registry = {}
        for doc in self.doc_registry_collection.find():
            if "_id" in doc:
                doc_id = str(doc.pop("_id"))  # Convert ObjectId to string
                registry[doc_id] = doc
        return registry
    
    def update_document_registry(self, registry: Dict[str, Any]):
        """Update the document registry in the database"""
        # Use upsert to create or update documents
        for doc_hash, info in registry.items():
            self.doc_registry_collection.update_one(
                {"_id": doc_hash},
                {"$set": info},
                upsert=True
            )


class ChatModel:
    """Handles the language model and chain operations"""
    
    def __init__(self, vector_manager: VectorStoreManager):
        """Initialize with a vector store manager"""
        self.vector_manager = vector_manager
        self.llm = ChatOpenAI(model=Config.GPT_MODEL)
    
    def _create_qa_chain(self):
        """Create the QA chain components"""
        # The retrieval prompt for standalone questions
        standalone_question_prompt = ChatPromptTemplate.from_messages([
            ("system", Config.STANDALONE_QUESTION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # The QA prompt for answering with context
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", Config.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Context: {context}"),
        ])
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        return standalone_question_prompt, document_chain
    
    def _convert_to_langchain_messages(self, mongo_messages: List[dict]) -> List[Any]:
        """Convert MongoDB message format to LangChain message objects"""
        chat_history = []
        for i in range(0, len(mongo_messages), 2):
            if i+1 < len(mongo_messages):
                if mongo_messages[i]["role"] == "user" and mongo_messages[i+1]["role"] == "assistant":
                    chat_history.append(HumanMessage(content=mongo_messages[i]["content"]))
                    chat_history.append(AIMessage(content=mongo_messages[i+1]["content"]))
        return chat_history
    
    def answer_question(self, session_id: str, question: str, chat_history: List[dict]) -> str:
        """Process a question and generate an answer using RAG"""
        # Get vectorstore for this session
        vectorstore = self.vector_manager.get_vectorstore_for_session(session_id)
        retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        
        # Convert history to LangChain format
        lc_history = self._convert_to_langchain_messages(chat_history)
        
        # Create chain components
        standalone_question_prompt, document_chain = self._create_qa_chain()
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=standalone_question_prompt
        )
        
        # Create the full chain
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )
        
        # Run the chain
        response = retrieval_chain.invoke({
            "input": question,
            "chat_history": lc_history
        })
        
        return response["answer"]


class ChatSession:
    """Represents a single chat session with a user"""
    
    def __init__(self, session_id: str, db_manager: DatabaseManager, chat_model: ChatModel):
        """Initialize a chat session"""
        self.session_id = session_id
        self.db_manager = db_manager
        self.chat_model = chat_model
    
    def process_question(self, question: str) -> str:
        """Process a user question and return an answer"""
        # Get chat history
        chat_history = self.db_manager.get_chat_history(self.session_id)
        
        # Generate answer
        answer = self.chat_model.answer_question(self.session_id, question, chat_history)
        
        # Save to history
        self.db_manager.add_message_pair(self.session_id, question, answer)
        
        return answer


class DocumentService:
    """Service for managing document operations"""
    
    def __init__(self, vector_manager: VectorStoreManager):
        """Initialize with vector store manager"""
        self.vector_manager = vector_manager
    
    def load_new_documents_from_folder(self, folder_path: str = Config.TXT_FOLDER) -> Dict[str, Any]:
        """Load new documents from a folder"""
        count = self.vector_manager.process_folder_for_new_documents(folder_path)
        return {
            "status": "success",
            "new_documents_processed": count,
            "message": f"Successfully processed {count} new documents"
        }
    
    def add_new_document(self, file_path: str, content: str) -> Dict[str, Any]:
        """Add a single new document"""
        count = self.vector_manager.process_new_document_content(file_path, content)
        return {
            "status": "success" if count > 0 else "no_action",
            "chunks_added": count,
            "message": f"Added {count} chunks to vectorstore" if count > 0 else "Document already processed or empty"
        }


class ChatServiceAPI:
    """API service layer managing chat operations"""
    
    def __init__(self):
        """Initialize service components"""
        self.db_manager = DatabaseManager()
        self.doc_registry = DocumentRegistry(self.db_manager)
        self.doc_loader = DocumentLoader(self.doc_registry)
        self.vector_manager = VectorStoreManager(self.doc_loader)
        self.chat_model = ChatModel(self.vector_manager)
        self.document_service = DocumentService(self.vector_manager)
    
    def create_new_session(self) -> str:
        """Create a new chat session"""
        return self.db_manager.create_new_session()
    
    def handle_chat_request(self, session_id: str, question: str) -> str:
        """Handle a chat request from a user"""
        session = ChatSession(session_id, self.db_manager, self.chat_model)
        return session.process_question(question)
    
    def load_new_documents(self, folder_path: str = None) -> Dict[str, Any]:
        """Load new documents from a folder"""
        return self.document_service.load_new_documents_from_folder(
            folder_path if folder_path else Config.TXT_FOLDER
        )
    
    def add_document(self, file_path: str, content: str) -> Dict[str, Any]:
        """Add a single document to the vectorstore"""
        return self.document_service.add_new_document(file_path, content)


# FastAPI Models
class ChatRequest(BaseModel):
    session_id: str
    question: str

class DocumentRequest(BaseModel):
    file_path: str
    content: str

class FolderRequest(BaseModel):
    folder_path: Optional[str] = None

# FastAPI Application Setup
app = FastAPI(title="Property Finder AI Chat API")
chat_service = ChatServiceAPI()


@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a chat message and return a response"""
    answer = chat_service.handle_chat_request(request.session_id, request.question)
    return {"answer": answer}


@app.get("/new_session")
async def new_session():
    """Create a new chat session"""
    session_id = chat_service.create_new_session()
    return {"session_id": session_id}


@app.post("/load_new_documents")
async def load_new_documents(request: FolderRequest = None):
    """Load new documents from a folder"""
    folder_path = request.folder_path if request and request.folder_path else None
    result = chat_service.load_new_documents(folder_path)
    return result


@app.post("/add_document")
async def add_document(request: DocumentRequest):
    """Add a single document to the vectorstore"""
    result = chat_service.add_document(request.file_path, request.content)
    return result


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document file"""
    # Read file content
    content = await file.read()
    content_str = content.decode("utf-8")
    
    # Use the filename as the file_path
    file_path = file.filename
    
    # Process the document
    result = chat_service.add_document(file_path, content_str)
    return result


# Entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)