# Code Slice Generation for Vulnerability Detection

This repository contains two Python scripts used for generating and utilizing code slices to analyze potential vulnerabilities, as described in our research paper.

## Overview

Our approach to slice construction involves the following steps:

1.  **Query Generation and Execution:** We use the SAST tool Joern and its CPGQL query language to identify potential vulnerability root causes by focusing on specific execution paths. This step is handled by the `generate_and_run_queries.py` script.
2.  **Slice Construction:** We construct the final code slice by gathering all code elements that influence both the identified execution path and its interacting variables. This results in a focused code snippet containing the essential components for vulnerability analysis. This step is handled by the `construct_slice` script.

## Scripts

### `generate_and_run_queries.py`

This script is responsible for:

* Generating CPGQL queries using a specified Large Language Model (LLM) such as DeepSeek or Qwen.
* Executing these generated queries against Joern servers to identify potential vulnerability root causes represented as execution paths within the code.

### Example Usage

Here's an example of how to run the `generate_and_run_queries.py` script with the essential arguments:

```bash
python generate_and_run_queries.py -d /path/to/your/dataset.json -o /path/to/your/output_dir --llm-endpoint [http://your.llm.host](http://your.llm.host):port
```
-d /path/to/your/dataset.json: Specifies the path to the JSON file containing the code samples to be processed.
-o /path/to/your/output_dir: Sets the base directory where the script will create results/ and logs/ subdirectories to store its output.
--llm-endpoint http://your.llm.host:port: Provides the endpoint (URL in this case) where the LLM service is running. Adjust the URL and port as needed.

*Optional Arguments:*

The script offers several optional arguments to customize its behavior.  For example, you can adjust the number of worker threads (-n), specify a different Docker Compose file for Joern (-c), or change the port used for the LLM (--llm-port).  To see a full list of available arguments and their descriptions, run the script with the -h or --help flag:

### Generate queries for your custom vulnerability dataset

To be able to use our script `generate_and_run_queries.py`, please follow these steps:
1. Create a json file with the same format as the files at `data/formai_query_generation.json` and `data/primevul_query_generation.json`.
2. Modify our Dockerfile: create your own `all_source_code.zip` file and modify the Dockerfile such that it uses your file and extracts the files at Joern's installation folder.


### `construct_slice.py`

This script takes the output from the query execution step (from generate_and_run_queries.py) and constructs code slices. It analyzes the identified execution paths and the variables interacting with them to produce focused code snippets relevant to the potential vulnerabilities.
Usage

### Example Usage

Here's an example of how to run the `construct_slice` script with a non-default dataset path:

```bash
python construct_slice -d /path/to/your/queries_output.json -o /path/to/your/output_dir.json
```

-d /path/to/your/queries_output.json: Specifies the path to the JSON file containing the output from the generate_and_run_queries.py script, which includes the identified vulnerability paths.
-o /path/to/your/output_dir: Sets the base directory where the script will create results/ and logs/ subdirectories to store its output.
Optional Arguments:

The script provides several optional arguments to customize its execution. For instance, you can adjust the number of Joern servers to use (-n), or provide a custom Docker Compose file (--docker-compose-file). To see a complete list of available arguments and their descriptions, run the script with the -h or --help flag
