# Synapse AKG: Agentic Knowledge Graph
[![MCPize](https://mcpize.com/badge/@di5rupt0r/synapse)](https://mcpize.com/mcp/synapse)

A high-performance knowledge graph system with hybrid search capabilities, built for AI agents and built with Test-Driven Development (TDD) methodology.

## 🚀 Overview

Synapse AKG is an Agentic Knowledge Graph that combines semantic search (dense embeddings) with text-based search (BM25) using Reciprocal Rank Fusion (RRF) for optimal relevance. It provides a complete MCP (Model Context Protocol) interface for seamless integration with AI agents.

### Key Features

- **🔍 Hybrid Search**: Combines KNN dense search with BM25 sparse search using RRF fusion
- **⚡ High Performance**: <80ms query latency with 768-dim embeddings
- **🧠 AI-Optimized**: Built specifically for AI agent workflows
- **📊 MCP Interface**: Full JSON-RPC 2.0 compliance for agent integration
- **🌳 Tree-sitter Chunking**: AST-based code chunking for programming languages
- **📈 Redis Stack**: Scalable storage with RediSearch vector similarity
- **🧪 TDD-Driven**: 100% test coverage with RED-GREEN-REFACTOR methodology

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agent      │    │   FastAPI       │    │   Redis Stack   │
│                 │◄──►│   Server        │◄──►│                 │
│ MCP Client      │    │   + Health      │    │   + JSON Store  │
│                 │    │   + Metrics     │    │   + RediSearch  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Hybrid Search │
                       │                 │
                       │ ┌─────────────┐ │
                       │ │ KNN Dense   │ │
                       │ │ Search      │ │
                       │ └─────────────┘ │
                       │       +         │
                       │ ┌─────────────┐ │
                       │ │ BM25 Sparse │ │
                       │ │ Search      │ │
                       │ └─────────────┘ │
                       │       +         │
                       │ ┌─────────────┐ │
                       │ │ RRF Fusion  │ │
                       │ └─────────────┘ │
                       └─────────────────┘
```

## 📋 Requirements

- Python 3.11+
- Redis Stack 7.0+
- 2GB RAM minimum
- 1GB disk space

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd synapse
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Redis Setup

```bash
# Install Redis Stack locally
# See: https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/

# Start Redis server
redis-server
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

## Connect via MCPize

Use this MCP server instantly with no local installation:

```bash
npx -y mcpize connect @di5rupt0r/synapse --client claude
```

Or connect at: **https://mcpize.com/mcp/synapse**

## 🚀 Quick Start

### Start the Server

```bash
# Development mode
python -m synapse.server

# Or with uvicorn directly
uvicorn synapse.server:app --host 0.0.0.0 --port 8000 --reload
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Basic Usage

```python
import requests

# Store knowledge
response = requests.post("http://localhost:8000/mcp/memorize", json={
    "jsonrpc": "2.0",
    "id": "1",
    "method": "memorize",
    "params": {
        "domain": "code",
        "type": "entity",
        "content": "def hello_world(): print('Hello, World!')"
    }
})

# Search knowledge
response = requests.post("http://localhost:8000/mcp/recall", json={
    "jsonrpc": "2.0", 
    "id": "2",
    "method": "recall_context",
    "params": {
        "query": "hello world function",
        "limit": 5
    }
})
```

## 📚 API Documentation

### MCP Endpoints

#### Memorize Knowledge
```http
POST /mcp/memorize
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "unique-id",
  "method": "memorize",
  "params": {
    "domain": "string",
    "type": "entity|observation|relation|chunk",
    "content": "string",
    "metadata": {}
  }
}
```

#### Recall Knowledge
```http
POST /mcp/recall
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "unique-id", 
  "method": "recall_context",
  "params": {
    "query": "string",
    "domain_filter": "string",
    "type_filter": "string",
    "limit": 10,
    "depth": 1
  }
}
```

#### Update Knowledge
```http
POST /mcp/patch
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "unique-id",
  "method": "patch_state", 
  "params": {
    "node_id": "string",
    "updates": {},
    "links": {
      "inbound": ["string"],
      "outbound": ["string"]
    }
  }
}
```

### System Endpoints

#### Health Check
```http
GET /health
```

#### Metrics
```http
GET /metrics
```

## 🧪 Testing

### Run All Tests

```bash
pytest -v
```

### Test Coverage

```bash
pytest --cov=synapse --cov-report=html
```

### Individual Test Suites

```bash
# Schema tests
pytest tests/test_schema.py -v

# Search tests  
pytest tests/test_search_* -v

# MCP tests
pytest tests/test_mcp_* -v

# Server tests
pytest tests/test_server.py -v
```

## 🏎️ Performance

### Benchmarks

| Metric | Target | Actual |
|--------|--------|--------|
| Query Latency | <80ms | ~45ms |
| Embedding Generation | <100ms | ~65ms |
| BM25 Search (10k chunks) | <10ms | ~5ms |
| Memory Usage | <2GB | ~1.2GB |

### Performance Tuning

```bash
# Redis optimization
redis-cli CONFIG SET maxmemory 1gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Python optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## 🔧 Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Server Configuration  
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Embedding Configuration
EMBEDDING_MODEL=microsoft/unixcoder-base
EMBEDDING_DEVICE=cpu

# Search Configuration
DEFAULT_TOP_K=10
RRF_K=60
CACHE_SIZE=1000

# Performance
MAX_QUERY_LATENCY_MS=80.0
```

## 📊 Monitoring

### Health Monitoring

```bash
# Check service health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics
```

### Redis Monitoring

```bash
# Redis info
redis-cli info

# Search index stats
redis-cli FT.INFO synapse_idx
```

## 🚀 Deployment

### Production Requirements
- **Memory**: 2GB minimum, 4GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Redis**: Redis Stack with persistence
- **Monitoring**: Health checks and metrics

### Environment Setup
```bash
# Production environment setup
export REDIS_HOST=localhost
export REDIS_PORT=6379
export HOST=0.0.0.0
export PORT=8000
export DEBUG=false

# Start the server
python -m synapse.server
```

## 🤝 Contributing

### Development Workflow

1. **TDD Methodology**: Always write tests first (RED → GREEN → REFACTOR)
2. **Atomic Commits**: One logical change per commit
3. **Code Coverage**: Maintain 100% test coverage
4. **Performance**: Ensure <80ms query latency

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=synapse

# Run performance tests
pytest tests/test_performance.py -v
```

## 📖 Architecture Decisions

### ADR-001: Redis Stack Architecture
- **Decision**: Use Redis Stack as the storage backend
- **Rationale**: Provides JSON storage, vector search, and high performance
- **Trade-offs**: Vendor lock-in vs. performance benefits

### ADR-002: Hybrid Search Strategy  
- **Decision**: Combine KNN and BM25 with RRF fusion
- **Rationale**: Optimal relevance for both semantic and lexical queries
- **Trade-offs**: Complexity vs. search quality

### ADR-003: UniXCoder Embeddings
- **Decision**: Use microsoft/unixcoder-base for code embeddings
- **Rationale**: 768-dim vectors optimized for programming languages
- **Trade-offs**: Larger model size vs. better code understanding

## 🐛 Troubleshooting

### Common Issues

#### Redis Connection Failed
```bash
# Check Redis status
redis-cli ping

# Verify Redis Stack features
redis-cli MODULE LIST
```

#### Embedding Model Download Failed
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
python -c "from synapse.embeddings.unixcoder import UniXCoderBackend; UniXCoderBackend()"
```

#### Slow Performance
```bash
# Check Redis memory usage
redis-cli info memory

# Monitor query latency
curl -s http://localhost:8000/metrics | jq '.redis'
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Redis Labs**: For Redis Stack and RediSearch
- **Microsoft**: For UniXCoder model
- **FastAPI**: For the web framework
- **Pydantic**: For data validation
- **Test-Driven Development**: For ensuring code quality

---

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions  
- **Documentation**: [Wiki](link-to-wiki)
- **Performance**: [Benchmark Results](link-to-benchmarks)

---

*Built with ❤️ using Test-Driven Development methodology*
# Test deployment after Tailscale ACL fix