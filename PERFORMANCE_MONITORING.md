# Performance Monitoring and Rollback Guide

This document provides guidelines for monitoring the performance of the hybrid PyMuPDF4LLM PDF extraction system and procedures for rolling back to the traditional extraction methods if performance issues arise.

## Overview

The hybrid extraction system uses PyMuPDF4LLM as the primary extraction method with automatic fallback to the traditional three-tier system (PyMuPDF → pdftotext → pdfminer.six). This approach provides enhanced heading detection and structured output while maintaining reliability through proven fallback mechanisms.

## Performance Monitoring

### Key Performance Metrics

Monitor these critical metrics to ensure the hybrid system performs optimally:

#### 1. Extraction Speed
- **Metric**: Time to extract text from PDF files
- **Baseline**: Traditional extraction baseline (measure before deployment)
- **Warning Threshold**: >150% of baseline extraction time
- **Critical Threshold**: >200% of baseline extraction time

#### 2. Memory Usage
- **Metric**: Peak memory consumption during extraction
- **Baseline**: Traditional extraction memory usage
- **Warning Threshold**: >150% of baseline memory usage
- **Critical Threshold**: >200% of baseline memory usage

#### 3. Success Rate
- **Metric**: Percentage of successful extractions
- **Baseline**: 95%+ success rate expected
- **Warning Threshold**: <90% success rate
- **Critical Threshold**: <85% success rate

#### 4. Quality Metrics
- **Text Length**: Amount of text extracted
- **Heading Detection**: Number of headings identified
- **Chunk Count**: Number of semantic chunks generated
- **Quality Score**: Overall extraction quality assessment

### Monitoring Implementation

#### Using the Benchmark Script

Run regular performance benchmarks using the provided script:

```bash
# Basic benchmark on sample files
python scripts/benchmark_extraction.py sample_book.pdf --output daily_benchmark.json

# Comprehensive benchmark with multiple iterations
python scripts/benchmark_extraction.py *.pdf --iterations 5 --output weekly_benchmark.json
```

#### Automated Monitoring

Set up automated monitoring by integrating benchmark runs into your CI/CD pipeline or cron jobs:

```bash
# Daily performance check (add to crontab)
0 2 * * * cd /path/to/project && python scripts/benchmark_extraction.py sample_book.pdf --quiet --output /var/log/pdf_performance.json
```

#### Log Analysis

Monitor application logs for performance indicators:

- PyMuPDF4LLM extraction failures
- Fallback activation frequency
- Quality assessment warnings
- Memory usage spikes
- Processing time anomalies

### Performance Thresholds

#### Green Zone (Normal Operation)
- Extraction time: ≤120% of baseline
- Memory usage: ≤130% of baseline
- Success rate: ≥95%
- Quality score: ≥0.7

#### Yellow Zone (Monitor Closely)
- Extraction time: 120-150% of baseline
- Memory usage: 130-150% of baseline
- Success rate: 90-95%
- Quality score: 0.5-0.7

#### Red Zone (Consider Rollback)
- Extraction time: >150% of baseline
- Memory usage: >150% of baseline
- Success rate: <90%
- Quality score: <0.5

## Warning Indicators

### Early Warning Signs

Watch for these indicators that may signal performance degradation:

1. **Increased Fallback Rate**
   - PyMuPDF4LLM failures becoming more frequent
   - Quality assessments consistently below threshold
   - More traditional extraction method usage

2. **Memory Pressure**
   - Gradual increase in peak memory usage
   - Memory allocation failures
   - System swapping during extraction

3. **Processing Delays**
   - Extraction times trending upward
   - User complaints about slow processing
   - Timeout errors in production

4. **Quality Degradation**
   - Reduced heading detection accuracy
   - Inconsistent text extraction
   - Semantic chunking issues

### Monitoring Commands

Use these commands to check system health:

```bash
# Check PyMuPDF4LLM availability and version
python -c "
from pdf_chunker.pymupdf4llm_integration import get_pymupdf4llm_info
import json
print(json.dumps(get_pymupdf4llm_info(), indent=2))
"

# Quick extraction test
python -c "
from pdf_chunker.core import process_document
import time
start = time.time()
result = process_document('sample_book.pdf', 8000, 200, ai_enrichment=False)
print(f'Extraction time: {time.time() - start:.2f}s')
print(f'Chunks generated: {len(result)}')
"

# Memory usage check
python -c "
import tracemalloc
tracemalloc.start()
from pdf_chunker.core import process_document
result = process_document('sample_book.pdf', 8000, 200, ai_enrichment=False)
current, peak = tracemalloc.get_traced_memory()
print(f'Peak memory: {peak / 1024 / 1024:.1f} MB')
tracemalloc.stop()
"
```

## Rollback Procedures

### When to Rollback

Consider rolling back to traditional extraction methods when:

1. **Performance Degradation**
   - Consistent performance metrics in the Red Zone
   - User experience significantly impacted
   - System stability compromised

2. **Reliability Issues**
   - Success rate drops below 85%
   - Frequent PyMuPDF4LLM crashes or errors
   - Data quality concerns

3. **Resource Constraints**
   - Memory usage exceeding system capacity
   - Processing times unacceptable for production
   - Infrastructure costs increasing significantly

### Rollback Methods

#### Method 1: Disable PyMuPDF4LLM Integration

**Temporary Disable (Environment Variable)**

Set an environment variable to disable PyMuPDF4LLM:

```bash
export DISABLE_PYMUPDF4LLM=true
```

Then modify the integration check in `pdf_chunker/pymupdf4llm_integration.py`:

```python
def is_pymupdf4llm_available() -> bool:
    """Check if PyMuPDF4LLM is available for use"""
    if os.environ.get('DISABLE_PYMUPDF4LLM', '').lower() in ('true', '1', 'yes'):
        return False
    return PYMUPDF4LLM_AVAILABLE and pymupdf4llm is not None
```

**Configuration-Based Disable**

Add a configuration option to disable PyMuPDF4LLM:

```python
# In your configuration file or settings
ENABLE_PYMUPDF4LLM = False

# In pdf_chunker/pymupdf4llm_integration.py
def is_pymupdf4llm_available() -> bool:
    """Check if PyMuPDF4LLM is available for use"""
    from your_config import ENABLE_PYMUPDF4LLM
    if not ENABLE_PYMUPDF4LLM:
        return False
    return PYMUPDF4LLM_AVAILABLE and pymupdf4llm is not None
```

#### Method 2: Adjust Quality Thresholds

Increase the quality threshold to force more fallbacks to traditional methods:

```python
# In pdf_chunker/pdf_parsing.py, modify the quality check
if quality_assessment['quality_score'] >= 0.9 and quality_assessment['has_content']:  # Raised from 0.6
    # Use PyMuPDF4LLM results
```

#### Method 3: Uninstall PyMuPDF4LLM

Complete removal of PyMuPDF4LLM dependency:

```bash
# Uninstall PyMuPDF4LLM
pip uninstall pymupdf4llm

# Remove from requirements.txt
sed -i '/pymupdf4llm/d' requirements.txt
```

#### Method 4: Code-Level Rollback

Modify the extraction function to skip PyMuPDF4LLM entirely:

```python
# In pdf_chunker/pdf_parsing.py, comment out or modify the PyMuPDF4LLM section
def extract_text_blocks_from_pdf(filepath: str, exclude_pages: str = None) -> list[dict]:
    """Extract structured text from a PDF using traditional methods only."""
    
    # Skip PyMuPDF4LLM extraction (rollback)
    # if is_pymupdf4llm_available():
    #     try:
    #         # PyMuPDF4LLM extraction code...
    #     except:
    #         # Fallback handling...
    
    # Proceed directly to traditional extraction
    doc = fitz.open(filepath)
    # ... rest of traditional extraction logic
```

### Rollback Validation

After implementing a rollback, validate the system:

1. **Run Benchmark Tests**
   ```bash
   python scripts/benchmark_extraction.py sample_book.pdf --output rollback_validation.json
   ```

2. **Check Performance Metrics**
   - Verify extraction times return to baseline
   - Confirm memory usage is within acceptable limits
   - Validate success rates meet expectations

3. **Quality Assessment**
   - Test heading detection with traditional methods
   - Verify text extraction completeness
   - Check semantic chunking consistency

4. **Integration Testing**
   - Run full integration validation tests
   - Verify all existing features still work
   - Test edge cases and error handling

### Rollback Communication

When rolling back, communicate clearly:

1. **Document the Decision**
   - Record performance metrics that triggered rollback
   - Note the specific rollback method used
   - Set timeline for re-evaluation

2. **Update Team**
   - Notify stakeholders of the rollback
   - Explain impact on functionality
   - Provide timeline for resolution

3. **Monitor Post-Rollback**
   - Continue performance monitoring
   - Track any functionality changes
   - Plan for future re-integration if appropriate

## Re-Integration Planning

### When to Re-Integrate

Consider re-enabling PyMuPDF4LLM when:

1. **Root Cause Resolved**
   - Performance issues identified and fixed
   - PyMuPDF4LLM updates address known problems
   - Infrastructure improvements support higher resource usage

2. **Improved Configuration**
   - Better quality thresholds identified
   - Optimized integration parameters
   - Enhanced fallback mechanisms

3. **Business Requirements**
   - Enhanced heading detection becomes critical
   - Structured output requirements increase
   - Competitive advantages justify performance trade-offs

### Re-Integration Process

1. **Gradual Rollout**
   - Start with non-critical workloads
   - Monitor performance closely
   - Gradually increase usage

2. **A/B Testing**
   - Run parallel systems for comparison
   - Measure performance differences
   - Validate quality improvements

3. **Staged Deployment**
   - Deploy to development environment first
   - Test in staging with production data
   - Monitor production deployment carefully

## Best Practices

### Regular Monitoring

- Run weekly performance benchmarks
- Monitor daily extraction metrics
- Set up automated alerts for threshold breaches
- Review performance trends monthly

### Documentation

- Keep detailed performance logs
- Document all configuration changes
- Maintain rollback decision records
- Update monitoring procedures regularly

### Testing

- Test rollback procedures regularly
- Validate monitoring systems
- Practice emergency response procedures
- Keep rollback scripts up to date

### Communication

- Establish clear escalation procedures
- Define roles and responsibilities
- Maintain stakeholder communication plans
- Document lessons learned

## Troubleshooting

### Common Issues

1. **PyMuPDF4LLM Import Errors**
   - Check installation: `pip list | grep pymupdf4llm`
   - Verify dependencies: `python -c "import pymupdf4llm"`
   - Reinstall if necessary: `pip install --force-reinstall pymupdf4llm`

2. **Memory Leaks**
   - Monitor memory usage over time
   - Check for unclosed file handles
   - Review PyMuPDF4LLM resource cleanup

3. **Performance Regression**
   - Compare with baseline metrics
   - Check for system resource constraints
   - Review recent configuration changes

4. **Quality Issues**
   - Validate input PDF files
   - Check PyMuPDF4LLM version compatibility
   - Review quality assessment thresholds

### Emergency Contacts

- **System Administrator**: [Contact Information]
- **Development Team Lead**: [Contact Information]
- **Performance Engineering**: [Contact Information]
- **On-Call Engineer**: [Contact Information]

## Conclusion

This monitoring and rollback guide ensures the hybrid PyMuPDF4LLM extraction system can be managed safely in production. Regular monitoring, clear thresholds, and well-tested rollback procedures provide confidence in deploying and maintaining this enhanced PDF extraction capability.

Remember: The goal is to leverage PyMuPDF4LLM's superior heading detection and structured output while maintaining the reliability and performance expectations of the existing system. When in doubt, prioritize system stability and user experience over new features.
