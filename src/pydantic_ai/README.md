# PydanticAI RnR Reason Code Classification

This directory contains a PydanticAI-based implementation for RnR (Return and Refund) Reason Code classification using AWS Bedrock. It's a rewrite of the original implementation using the PydanticAI framework for better structured output, type safety, and async support.

## Features

- **PydanticAI Framework**: Uses PydanticAI for structured output and automatic validation
- **AWS Bedrock Integration**: Supports both standard models and inference profiles
- **Async Support**: Native async/await support for better performance
- **Type Safety**: Full Pydantic model validation for inputs and outputs
- **Inference Profile Support**: Compatible with Claude 4 and other provisioned throughput models
- **Robust Error Handling**: Comprehensive fallback mechanisms
- **Batch Processing**: Efficient batch processing with concurrent execution
- **Multiple Output Formats**: Saves results in Parquet, CSV, and JSONL formats

## Files

### Core Implementation
- `rnr_reason_code_models.py` - Pydantic models for data validation and structured output
- `bedrock_rnr_agent.py` - PydanticAI agent for RnR classification using AWS Bedrock
- `rnr_bedrock_main.py` - Main processor with batch processing capabilities

### Testing
- `../test/pydantic_ai/test_pydantic_rnr.py` - Comprehensive pytest test suite

## Models and Categories

The implementation classifies RnR cases into 12 categories:

1. **TrueDNR** - Package delivered but buyer claims non-receipt
2. **Confirmed_Delay** - Shipment delayed due to external factors
3. **Delivery_Attempt_Failed** - Unsuccessful delivery attempt
4. **Seller_Unable_To_Ship** - Seller cannot fulfill order
5. **PDA_Undeliverable** - Item stuck in transit without confirmation
6. **PDA_Early_Refund** - Refund given before delivery
7. **Buyer_Received_WrongORDefective_Item** - Quality/condition issues
8. **Returnless_Refund** - Refund without return requirement
9. **BuyerCancellation** - Buyer cancels before delivery
10. **Return_NoLongerNeeded** - Post-delivery return of unwanted items
11. **Product_Information_Support** - Information requests only
12. **Insufficient_Information** - Missing context for classification

## Usage

### Basic Usage

```python
from src.pydantic_ai.bedrock_rnr_agent import BedrockRnRPydanticAgent

# Initialize the agent
agent = BedrockRnRPydanticAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-west-2",
    use_inference_profile=True
)

# Analyze a single case
result = agent.analyze_rnr_case_sync(
    dialogue="[BUYER]: I haven't received my package...",
    shiptrack="[Event Time]: 2025-02-21T17:40:49.323Z [Ship Track Event]: Delivered to customer",
    max_estimated_arrival_date="2025-02-22"
)

print(f"Category: {result.category}")
print(f"Confidence: {result.confidence_score}")
print(result.to_formatted_output())
```

### Async Usage

```python
import asyncio

async def analyze_case():
    agent = BedrockRnRPydanticAgent(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-west-2"
    )
    
    result = await agent.analyze_rnr_case_async(
        dialogue="[BUYER]: I want to cancel my order...",
        shiptrack="",
        max_estimated_arrival_date=None
    )
    
    return result

# Run async
result = asyncio.run(analyze_case())
```

### Batch Processing

```python
from src.pydantic_ai.rnr_bedrock_main import RnRPydanticBedrockProcessor
import pandas as pd

# Initialize processor
processor = RnRPydanticBedrockProcessor(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-west-2",
    output_dir="./output"
)

# Load data
df = pd.read_csv("input_data.csv")  # Must have 'dialogue' and 'shiptrack' columns
df = processor.validate_input_data(df)

# Process batch (async by default for better performance)
results_df = processor.process_batch(
    df=df,
    batch_size=10,
    use_async=True
)

# Save results
output_files = processor.save_results(results_df)
print(f"Results saved to: {output_files}")
```

### Command Line Usage

```bash
python -m src.pydantic_ai.rnr_bedrock_main \
    --input-file data.csv \
    --output-dir ./output \
    --model-id anthropic.claude-3-5-sonnet-20241022-v2:0 \
    --region-name us-west-2 \
    --batch-size 10 \
    --use-async
```

## Inference Profile Support

For Claude 4 and other models requiring provisioned throughput:

### Using Environment Variables

```bash
export BEDROCK_INFERENCE_PROFILE_ARN="arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-profile"
```

### Using Profile ARN

```python
agent = BedrockRnRPydanticAgent(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    inference_profile_arn="arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-profile",
    use_inference_profile=True
)
```

### Using Global Profile ID

```python
agent = BedrockRnRPydanticAgent(
    model_id="global.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    use_inference_profile=True
)
```

## Input Data Format

Input data should be a CSV, JSON, JSONL, or Parquet file with the following columns:

- `dialogue` (required): Buyer-seller conversation messages
- `shiptrack` (required): Ship track events history
- `max_estimated_arrival_date` (optional): Estimated delivery date

Example CSV:
```csv
dialogue,shiptrack,max_estimated_arrival_date
"[BUYER]: Package not received [SELLER]: Let me check tracking","[Event Time]: 2025-02-21T17:40:49.323Z [Ship Track Event]: Delivered",2025-02-22
"[BUYER]: Want to cancel order [SELLER]: No problem","",
```

## Output Format

The implementation returns structured `RnRReasonCodeAnalysis` objects with:

- `category`: One of the 12 predefined categories
- `confidence_score`: Decimal between 0.00 and 1.00
- `key_evidence`: Message, shipping, and timeline evidence
- `reasoning`: Primary factors, supporting evidence, and contradicting evidence

### Formatted Output Example

```
1. Category: TrueDNR

2. Confidence Score: 0.92

3. Key Evidence:
   * Message Evidence:
     [sep] [BUYER]: Hello, I have not received my package, but I see the order shows that it has been delivered, why?
     [sep] [BUYER]: But I did not find any package, please refund me, thank you
   * Shipping Evidence:
     [sep] [Event Time]: 2025-02-21T17:40:49.323Z [Ship Track Event]: Delivered to customer
     [sep] No further shipping events after delivery confirmation
   * Timeline Evidence:
     [sep] Delivery confirmation on 2025-02-21 17:40
     [sep] Buyer reports non-receipt starting 2025-02-25 07:14

4. Reasoning:
   * Primary Factors:
     [sep] Tracking shows package was delivered successfully
     [sep] Buyer explicitly states they did not receive the package after delivery scan
   * Supporting Evidence:
     [sep] Buyer requests refund due to missing package
     [sep] No evidence of buyer receiving wrong/defective item
   * Contradicting Evidence:
     [sep] None
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest test/pydantic_ai/test_pydantic_rnr.py -v

# Run specific test class
pytest test/pydantic_ai/test_pydantic_rnr.py::TestPydanticModels -v

# Run with coverage
pytest test/pydantic_ai/test_pydantic_rnr.py --cov=src.pydantic_ai -v
```

## Comparison with Original Implementation

| Feature | Original (src/bedrock) | PydanticAI (src/pydantic_ai) |
|---------|----------------------|------------------------------|
| Framework | Direct boto3 calls | PydanticAI with structured output |
| Type Safety | Manual validation | Automatic Pydantic validation |
| Async Support | Manual event loops | Native async/await |
| Error Handling | Basic try/catch | Comprehensive fallback mechanisms |
| Output Validation | Manual parsing | Automatic model validation |
| Batch Processing | Sequential | Concurrent async processing |
| Testing | Basic tests | Comprehensive pytest suite |

## Dependencies

- `pydantic-ai` - PydanticAI framework
- `pydantic` - Data validation and settings management
- `boto3` - AWS SDK for Python
- `pandas` - Data manipulation and analysis
- `tenacity` - Retry library
- `asyncio` - Asynchronous I/O support

## Configuration

### Model Compatibility

**Inference Profile Required:**
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `anthropic.claude-4-*` models
- `anthropic.
