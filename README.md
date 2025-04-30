# Email and Message Text Processing Pipeline

A general email and message processing pipeline that handles email exchanges and text messages. 

This text pipeline is used as a pre-processing pipeline for a classification system based on an encoder-only Masked Language Model (MLM).

## Features
- Handles email threads and text message formatting.
- Prepares text data for downstream NLP tasks such as classification.
- Supports tokenization, cleaning, and feature extraction.

## Code Structure
The repository is organized as follows:

- **`data/`**: Sample data and datasets for testing.
- **`preprocessing/`**: Preprocessing modules (e.g., cleaning, tokenization).
- **`models/`**: Model-related utilities and configurations.
- **`tests/`**: Unit tests for the pipeline.
- **`utils/`**: Helper functions and utilities.
- **`main.py`**: Entry point for running the pipeline.
- **`README.md`**: Documentation.

## Installation
To use this pipeline, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/nlp-pipeline.git
cd nlp-pipeline
pip install -r requirements.txt
```

## Usage
You can run the pipeline using the main.py script. Below is an example of how to use it:

```bash
python main.py --input data/sample_emails.txt --output results/processed_output.txt
```

Command-line Arguments
- `--input`: Path to the input file containing raw email or message data.
- `--output`: Path to save the processed output.
- `--config`: (Optional) Path to a configuration file for custom preprocessing settings.

## Testing
To ensure everything is working correctly, run the unit tests:

```bash
pytest tests/
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details. 
