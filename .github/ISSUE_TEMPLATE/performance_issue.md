---
name: Performance Issue
about: Report performance problems or bottlenecks
title: '[PERFORMANCE] '
labels: performance
assignees: ''
---

## ⚡ Performance Issue Description
Describe the performance problem you're experiencing.

## 📊 Performance Metrics
Please provide specific performance measurements:

### Query Performance
- **Query Type**: [e.g. hybrid search, BM25, KNN]
- **Expected Latency**: [e.g. <80ms]
- **Actual Latency**: [e.g. 250ms]
- **Dataset Size**: [e.g. 10k chunks, 100k vectors]

### System Resources
- **CPU Usage**: [e.g. 80% during query]
- **Memory Usage**: [e.g. 2GB total, 1.5GB for embeddings]
- **Redis Memory**: [e.g. 500MB used]

## 🔄 Reproduction Steps
Steps to reproduce the performance issue:
1. Set up dataset with [size] chunks
2. Run query: [query example]
3. Measure latency: [measurement method]
4. Observe: [observed behavior]

## 🎯 Expected Performance
Based on the project targets:
- **Hybrid Search**: <80ms
- **BM25 Search**: <10ms for 10k chunks
- **KNN Search**: <50ms for 10k vectors
- **Graph Resolution**: <10ms

## 🖥️ Environment Details
- **Hardware**: [CPU, RAM, storage]
- **OS**: [Operating system and version]
- **Redis Version**: [Redis Stack version]
- **Python Version**: [Python version]
- **Concurrent Load**: [number of concurrent requests]

## 📈 Profiling Data
If available, please provide profiling data:
```bash
# Example profiling commands
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/mcp/recall" -d '{"jsonrpc":"2.0","id":"1","method":"recall_context","params":{"query":"test"}}'

# Redis slowlog
redis-cli SLOWLOG GET 10
```

## 🔍 Analysis
What do you believe is causing the performance bottleneck?
- [ ] Redis query optimization
- [ ] Embedding generation
- [ ] BM25 indexing
- [ ] RRF fusion algorithm
- [ ] Graph resolution
- [ ] Network latency
- [ ] Memory allocation

## 💡 Proposed Solutions
Any suggestions for performance improvements:

## ✅ Checklist
- [ ] I have provided specific performance measurements
- [ ] I have included environment details
- [ ] I have measured against project targets
- [ ] I have considered profiling data
