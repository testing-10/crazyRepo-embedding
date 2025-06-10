# Embedding Model Testing Framework

A comprehensive, modular testing framework for evaluating embedding models across semantic similarity, retrieval, clustering, and efficiency metrics. Designed to support downstream AI-powered, RAG-based PowerPoint generation systems.

## Features

- **Multi-Model Support**: BGE-M3, E5-Mistral, OpenAI Embeddings, Cohere Embed v3, Jina Embeddings, GTE-Large, Sentence-Transformers
- **Comprehensive Evaluation**: Semantic similarity, retrieval, clustering, classification, efficiency, and robustness testing
- **Configuration-Driven**: YAML-based execution with model-specific configs
- **Rich Reporting**: HTML, JSON, and CSV outputs with visualizations
- **Cost Tracking**: Monitor API usage and costs across providers
- **Modular Architecture**: Easy to extend with new models, metrics, and evaluators

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```bash
python embedding_executor.py --config configs/default_config.yaml
```

## Project Structure

```
Directory structure:
└── testing-10-crazyrepo-embedding/
    ├── README.md
    ├── embedding_client_factory.py                  # Unit and integration tests
    ├── embedding_executor.py
    ├── requirements.txt
    ├── test_embedding_imports.py
    ├── test_embedding_integration.py
    ├── clients/                                     # Model client implementations
    │   ├── __init__.py
    │   ├── azure_embedding_client.py
    │   ├── base_embedding_client.py
    │   ├── cohere_embedding_client.py
    │   ├── huggingface_embedding_client.py
    │   ├── jina_embedding_client.py
    │   ├── openai_embedding_client.py
    │   └── sentence_transformer_client.py
    ├── configs/                                     # Configuration files
    │   ├── embedding_test_config.yaml
    │   └── models/
    │       ├── azure_embeddings.yaml
    │       ├── cohere_embeddings.yaml
    │       ├── huggingface_embeddings.yaml
    │       ├── jina_embeddings.yaml
    │       ├── openai_embeddings.yaml
    │       └── sentence_transformers.yaml
    ├── costs/
    │   └── embedding_api_costs.json
    ├── datasets/                                    # Dataset loaders
    │   ├── __init__.py
    │   ├── dataset_loader.py
    │   ├── benchmark_datasets/
    │   │   ├── __init__.py
    │   │   ├── beir_datasets.py
    │   │   ├── msmarco.py
    │   │   ├── nfcorpus.py
    │   │   ├── quora_duplicates.py
    │   │   └── sts_benchmark.py
    │   └── custom_datasets/
    │       ├── code_snippets.json
    │       ├── domain_specific_texts.json
    │       ├── legal_documents.json
    │       └── scientific_papers.json
    ├── evaluators/                                   # Evaluation logic
    │   ├── __init__.py
    │   ├── base_embedding_evaluator.py
    │   ├── classification_evaluator.py
    │   ├── clustering_evaluator.py
    │   ├── efficiency_evaluator.py
    │   ├── evaluator_factory.py
    │   ├── retrieval_evaluator.py
    │   ├── robustness_evaluator.py
    │   └── semantic_similarity_evaluator.py
    ├── metrics/                                      # Metric calculations
    │   ├── __init__.py
    │   ├── classification_metrics.py
    │   ├── clustering_metrics.py
    │   ├── efficiency_metrics.py
    │   ├── retrieval_metrics.py
    │   └── similarity_metrics.py
    ├── reporting/                                    # Report generation
    │   ├── __init__.py
    │   ├── dimension_analysis.py
    │   ├── embedding_comparison_report.py
    │   └── embedding_visualizations.py
    ├── test_cases/                                   # Test datasets
    │   ├── classification/
    │   │   ├── intent_classification.json
    │   │   ├── sentiment_analysis.json
    │   │   └── text_classification.json
    │   ├── clustering/
    │   │   ├── document_clustering.json
    │   │   ├── hierarchical_clustering.json
    │   │   └── topic_clustering.json
    │   ├── efficiency/
    │   │   ├── batch_processing.json
    │   │   ├── memory_usage.json
    │   │   └── speed_benchmark.json
    │   ├── retrieval/
    │   │   ├── code_search.json
    │   │   ├── cross_modal_search.json
    │   │   ├── document_search.json
    │   │   └── qa_retrieval.json
    │   ├── robustness/
    │   │   ├── adversarial_examples.json
    │   │   ├── length_variation.json
    │   │   └── noise_resistance.json
    │   └── semantic_similarity/
    │       ├── cross_lingual.json
    │       ├── domain_adaptation.json
    │       ├── paragraph_similarity.json
    │       └── sentence_pairs.json
    └── utils/                                   # Core utilities
        ├── __init__.py
        ├── embedding_cost_tracker.py
        ├── embedding_file_utils.py
        ├── embedding_logger.py
        └── vector_operations.py
```

## Configuration

### Model Configuration

Add new models in `configs/model_configs/`:

```yaml
model_name: "custom-model"
provider: "custom"
config:
  api_key_env: "CUSTOM_API_KEY"
  model_id: "custom-model-v1"
```

### Evaluation Configuration

Customize evaluations in `configs/evaluation_configs/`:

```yaml
evaluators:
  - name: "semantic_similarity"
    enabled: true
    config:
      similarity_threshold: 0.7
```

## Supported Models

| Provider | Model | Dimensions | Context Length |
|----------|-------|------------|----------------|
| BAAI | BGE-M3 | 1024 | 8192 |
| Microsoft | E5-Mistral-7B | 4096 | 32768 |
| OpenAI | text-embedding-3-large | 3072 | 8191 |
| Cohere | embed-english-v3.0 | 1024 | 512 |
| Jina | jina-embeddings-v2 | 768 | 8192 |
| Alibaba | GTE-Large | 1024 | 512 |

## Evaluation Categories

### Semantic Similarity
- Cosine similarity on sentence pairs
- Spearman correlation with human judgments
- Cross-lingual similarity assessment

### Retrieval
- Precision@K, Recall@K, F1@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

### Clustering
- Silhouette score, Calinski-Harabasz index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

### Classification
- Accuracy, Precision, Recall, F1-score
- ROC-AUC for binary classification
- Macro/micro averages for multi-class

### Efficiency
- Embedding generation time
- Memory usage tracking
- Throughput measurements

### Robustness
- Performance under noise
- Adversarial input handling
- Cross-domain generalization

## Reporting

The framework generates comprehensive reports including:

- **Performance Comparisons**: Side-by-side model comparisons
- **Visualizations**: Scatter plots, heatmaps, performance charts
- **Cost Analysis**: API usage and cost breakdowns
- **Dimension Analysis**: Embedding space characteristics

### Report Formats

- **HTML**: Interactive reports with charts and tables
- **JSON**: Machine-readable results for further processing
- **CSV**: Tabular data for spreadsheet analysis

## Testing

Run the test suite:

```bash
# Import validation
python tests/test_embedding_imports.py

# Integration tests
python tests/test_embedding_integration.py
```

## Adding New Components

### New Embedding Model

1. Create client in `src/embedding_clients/`
2. Add configuration in `configs/model_configs/`
3. Update factory in `embedding_client_factory.py`

### New Evaluator

1. Extend `BaseEmbeddingEvaluator` in `src/evaluators/`
2. Implement required methods
3. Add to evaluation configuration

### New Metric

1. Create metric function in `src/metrics/`
2. Follow naming convention: `calculate_metric_name()`
3. Include proper error handling and validation

## Logging

Logs are written to `logs/` directory with different levels:
- `INFO`: General execution flow
- `DEBUG`: Detailed debugging information
- `ERROR`: Error conditions and exceptions
- `WARNING`: Non-critical issues

## Cost Tracking

Monitor API costs in real-time:
- Per-model cost breakdown
- Token usage statistics
- Cost projections and budgeting

## Contributing

1. Follow the modular architecture patterns
2. Include comprehensive error handling
3. Add appropriate logging statements
4. Update tests for new functionality
5. Document configuration options

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the logs in `logs/` directory
- Review configuration files
- Run import validation tests
- Contact the development team
