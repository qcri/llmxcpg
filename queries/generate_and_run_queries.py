import json
import os
import sys
import threading
import time
import argparse
import logging
from typing import List, Dict
from Components.model import LLMManager, Model
from Components.joern_manager import JoernManager

# --- Configuration for Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def gen_query_prompt(code: str):
    instruction = """Your task is to design Precise Joern CPGQL Queries for Vulnerability Analysis.

Objective:
Develop targeted CPGQL Joern queries to:
1. Identify taint flows based on your analysis
2. Capture potential vulnerability paths

Constraints:
- Queries must be executable in Joern/CPGQL
- Use Scala language features for query construction
- Last query must use reachableByFlows to identify vulnerable paths

Output Requirements:
- Provide a JSON object with one field:
  "queries": Sequence of CPGQL queries to detect vulnerability

Expected JSON Output Format:
```json
{
  "queries": ["Query1", "Query2", ..., "Final Reachable Flows Query"]
}
```

Example Output:
```json
{
  "queries": [
    "val freeCallsWithIdentifier = cpg.method.name("(.*_)?free").filter(_.parameter.size == 1).callIn.where(_.argument(1).isIdentifier).l",
    "freeCallsWithIdentifier.flatMap(f => {val freedIdentifierCode = f.argument(1).code; val postDom = f.postDominatedBy.toSetImmutable; val assignedPostDom = postDom.isIdentifier.where(_.inAssignment).codeExact(freedIdentifierCode).flatMap(id => id ++ id.postDominatedBy); postDom.removedAll(assignedPostDom).isIdentifier.codeExact(freedIdentifierCode).reachableByFlows(f.argument(1))}).l"
  ]
}
```
""" 
    prompt = f"{instruction}\nInput:\n{code}"
    return prompt

class DatasetProcessor:
    """
    Processes each datapoint, generates the Joern queries for this specific sample, and finally executes the joern queries
    to extract the potentially vulnerable execution paths.
    Each instance runs in its own thread.
    """
    def __init__(self,
                 port: int,
                 dataset_slice: List[Dict],
                 output_file: str,
                 logs_file: str,
                 compose_file: str,
                 llm_model_type: str,
                 llm_model_name: str,
                 llm_port: int,
                 joern_recreate_interval: int):
        """
        Initialize a dataset processor for a specific port and dataset slice.

        Args:
            port: Joern server port number for this processor.
            dataset_slice: Subset of the dataset to process.
            output_file: Path to the unique output file for this thread's results.
            logs_file: Path to the file to store detailed logs/errors for this thread.
            compose_file: Path to the Docker compose file for Joern.
            llm_model_type: Identifier string for the LLM model type (e.g., "DeepSeek").
            llm_model_name: Model identifier passed to the completion API (e.g., local model name).
            llm_port: Port for the LLM service.
            joern_recreate_interval: Number of samples to process before recreating Joern server.
        """
        self.port = port
        self.dataset_slice = dataset_slice
        self.output_file = output_file
        self.logs_file = logs_file
        self.compose_file = compose_file
        self.joern_recreate_interval = joern_recreate_interval
        self.llm_model_type = llm_model_type
        self.llm_model_name = llm_model_name
        self.llm_port = llm_port

        self.current_sample_uuid = ""
        self.sample_log_buffer = []

        # Ensure output/log directories exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.logs_file), exist_ok=True)

    def _log_sample_message(self, level: int, message: str, **kwargs):
        """Logs a message and adds it to the sample buffer."""
        log_msg = f"[Port {self.port} | Sample {self.current_sample_uuid}] {message}"
        logging.log(level, log_msg, **kwargs)
        self.sample_log_buffer.append(f"[{logging.getLevelName(level)}] {message}")

    def process_dataset(self):
        """
        Process the assigned slice of the dataset. Handles Joern server recreation.
        """
        # Initialize thread-specific asyncio settings
        import nest_asyncio
        import asyncio
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Apply nest_asyncio in this thread's context
        nest_asyncio.apply(loop)

        # Initialize components (intentionally after the event loop is setup)
        self.joern_manager = JoernManager(self.port, self.compose_file)

        active_joern_project = None # Track the currently loaded project filename
        self.sample_log_buffer = []  # Ensure log buffer exists for error handling
        try:
            model_type_map = {
                "deepseek": Model.DEEPSEEK,
                "vllm": Model.VLLM,
            }
            model_type = model_type_map.get(self.llm_model_type.lower())
            if model_type is None:
                raise ValueError(f"Unsupported llm_model_type: {self.llm_model_type}")

            self.llm_manager = LLMManager(model_type, self.llm_model_name, port=self.llm_port)
            num_samples = len(self.dataset_slice)
            for i, sample in enumerate(self.dataset_slice):
                self.current_sample_uuid = sample.get("uuid", "N/A")
                self.sample_log_buffer = [] # Reset buffer for new sample
                active_joern_project = None # Reset for each sample initially

                # --- Joern Server Recreation Logic ---
                if i > 0 and i % self.joern_recreate_interval == 0:
                    # Clean up any project loaded *before* recreating server
                    if active_joern_project:
                         try:
                             self._log_sample_message(logging.DEBUG, f"Cleaning up project '{active_joern_project}' before Joern recreate.")
                             self.joern_manager.delete_project(active_joern_project)
                         except Exception as cleanup_e:
                             self._log_sample_message(logging.WARNING, f"Failed to cleanup Joern project {active_joern_project} before recreate: {cleanup_e}")
                         finally:
                            active_joern_project = None # Mark as cleaned up

                    self._log_sample_message(logging.INFO, f"Processed {self.joern_recreate_interval} samples. Recreating Joern server.")

                    is_healthy = self.joern_manager.recreate_server()
                    if not is_healthy:
                        self._log_sample_message(logging.ERROR, "Joern server unhealthy after recreation. Exiting thread.")
                        self._write_error_logs()
                        return # Exit thread
                    self._log_sample_message(logging.INFO, "Joern server recreated successfully.")
                # --- End Joern Server Recreation Logic ---

                try:
                    self._log_sample_message(logging.INFO, f"Starting processing.")
                    file_name = sample.get("file_name", "").split("/")[-1]
                    if not file_name:
                         raise ValueError("Could not extract a valid file_name from sample['file_name']")

                    # Process this sample
                    result = self._process_single_sample(sample, file_name)
                    if result:
                        active_joern_project = file_name # Mark project as potentially loaded
                        self._write_processed_sample(result)
                        self._log_sample_message(logging.INFO, f"Successfully processed and saved.")
                    else:
                        # Error occurred within _process_single_sample, logs already added
                        self._log_sample_message(logging.WARNING, f"Processing failed. Check logs.")
                        # Don't write results, but ensure logs are written
                        self._write_error_logs()
                        # If processing failed, we might still have loaded the project
                        active_joern_project = file_name

                except Exception as e:
                    # Catch errors during the loop iteration (e.g., unexpected issues)
                    self._log_sample_message(logging.ERROR, f"Unhandled error processing sample: {e}", exc_info=True)
                    self._write_error_logs() # Write error details
                    # Mark project as potentially active if filename was determined
                    if 'file_name' in locals() and file_name:
                         active_joern_project = file_name

                finally:
                    # Clean up Joern project *after* processing or error for this sample
                    if active_joern_project:
                         try:
                             self._log_sample_message(logging.DEBUG, f"Cleaning up Joern project: {active_joern_project}")
                             self.joern_manager.delete_project(active_joern_project)
                         except Exception as cleanup_e:
                             self._log_sample_message(logging.WARNING, f"Failed to cleanup Joern project {active_joern_project}: {cleanup_e}")
                         finally:
                             active_joern_project = None # Reset tracker


        except Exception as e:
            # Catch critical errors during thread execution
            logging.error(f"[Port {self.port}] Critical error in thread execution: {e}", exc_info=True)
            self.sample_log_buffer.append(f"[CRITICAL] Thread execution error: {e}")
            self._write_error_logs()
        finally:
            # Close the event loop when done
            loop.close()


    def _process_single_sample(self, sample: Dict, file_name: str) -> Dict | None:
        """
        Process a single sample using Joern and LLM.

        Args:
            sample: The sample dictionary to process.
            file_name: The extracted filename for Joern operations.

        Returns:
            The processed sample result as a dictionary, or None if processing failed.
        """
        try:
            # --- Validate Sample ---
            required_keys = ["file_name", "code", "label"]
            if not all(key in sample for key in required_keys):
                 missing = [key for key in required_keys if key not in sample]
                 raise ValueError(f"Sample missing required keys: {missing}")
            code_content = sample["code"] # Get code content


            # --- Load code into Joern ---
            self._log_sample_message(logging.DEBUG, f"Loading project '{file_name}' into Joern.")
            stdout = self.joern_manager.load_project(file_name)

            # Basic error check based on original code
            if "io.joern.console.ConsoleException" in stdout:
                 self._log_sample_message(logging.ERROR, f"Joern ConsoleException loading project '{file_name}': {stdout}")
                 raise Exception(f"Joern ConsoleException loading project: {stdout}")
            elif "fail" in stdout.lower() or "error" in stdout.lower():
                 self._log_sample_message(logging.WARNING, f"Potential issue loading project '{file_name}' in Joern: {stdout}")


            # --- Generate CPGQL queries using LLM ---
            self._log_sample_message(logging.DEBUG, "Generating CPGQL queries using LLM.")
            prompt = gen_query_prompt(code_content)
            message_history = [{"role": "user", "content": prompt}]

            llm_response = self.llm_manager.send_messages(message_history)
            if not llm_response:
                raise Exception("LLM response was empty or invalid.")

            completion_text = self.llm_manager.get_completion_text(llm_response)
            if not completion_text:
                 raise Exception("Failed to extract completion text from LLM response.")

            llm_answer = self.llm_manager.extract_queries(completion_text)
            if not llm_answer or "queries" not in llm_answer or not isinstance(llm_answer["queries"], list):
                 self._log_sample_message(logging.WARNING, f"Failed to extract valid list of queries from LLM response. Response text: {completion_text}")
                 raise Exception("Failed to extract valid queries list from LLM response.")

            generated_queries = llm_answer["queries"]
            self._log_sample_message(logging.INFO, f"LLM generated {len(generated_queries)} queries.")
            self._log_sample_message(logging.DEBUG, f"Generated Queries: {generated_queries}")


            # --- Run queries and extract paths ---
            self._log_sample_message(logging.DEBUG, "Running generated queries in Joern.")
            successful, paths = self.joern_manager.run_queries(generated_queries, code_content)

            self._log_sample_message(logging.INFO, f"Joern query validation: successful={successful}")
            self._log_sample_message(logging.DEBUG, f"Extracted paths: {paths}")


            # --- Create result dictionary ---
            sample_result = {
                "file_name": sample["file_name"],
                "llm_model_type": self.llm_model_type,
                "llm_queries": generated_queries,
                "label": sample["label"],
                "joern_results": {
                     "all_paths": paths,
                     "successful_query_validation": successful
                },
                "processing_status": "success",
                "details": sample
            }

            return sample_result

        except Exception as e:
            self._log_sample_message(logging.ERROR, f"Error processing sample: {e}", exc_info=True)
            # No explicit cleanup here, handled by finally block in process_dataset
            return None # Indicate failure

    def _write_processed_sample(self, processed_sample: Dict):
        """Appends a processed sample to the thread-specific JSON output file."""
        try:
            # Read existing data if file exists, otherwise start fresh
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    self._log_sample_message(logging.WARNING, f"Output file {self.output_file} does not contain a JSON list. Overwriting.")
                    existing_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            # Append new sample
            existing_data.append(processed_sample)

            # Write back to file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)

            logging.info(f"[Port {self.port}] Appended result for sample {self.current_sample_uuid} to {self.output_file}")

        except Exception as e:
             logging.error(f"[Port {self.port} | Sample {self.current_sample_uuid}] Error writing processed sample to {self.output_file}: {e}", exc_info=True)
             self.sample_log_buffer.append(f"[ERROR] Failed to write result to output file: {e}")
             self._write_error_logs() # Attempt to write the error log mentioning this failure


    def _write_error_logs(self):
        """Appends the buffered logs for the current sample to the thread's log file."""
        if not self.sample_log_buffer:
            return

        try:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "port": self.port,
                "sample_uuid": self.current_sample_uuid,
                "log_messages": self.sample_log_buffer
            }

            # Read existing logs if file exists
            try:
                with open(self.logs_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
                if not isinstance(existing_logs, list):
                     logging.warning(f"Log file {self.logs_file} corrupt. Starting new list.")
                     existing_logs = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_logs = []

            # Append new log entry
            existing_logs.append(log_entry)

            # Write back to file
            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, indent=4)

            logging.info(f"[Port {self.port}] Appended logs for sample {self.current_sample_uuid} to {self.logs_file}")

        except Exception as e:
            logging.critical(f"[Port {self.port} | Sample {self.current_sample_uuid}] CRITICAL ERROR: Failed to write to error log file {self.logs_file}: {e}", exc_info=True)
        finally:
             # Clear buffer regardless of write success/failure
             self.sample_log_buffer = []


def main():
    """
    Main function to parse arguments, set up, and run dataset processing threads.
    """
    parser = argparse.ArgumentParser(description="Process a dataset of code samples using Joern and an LLM in parallel.")

    # --- Command Line Arguments ---
    parser.add_argument("-d", "--dataset-file", type=str, required=True,
                        help="Path to the input dataset JSON file.")
    parser.add_argument("-o", "--output-base-dir", type=str, required=True,
                        help="Base directory for output files. 'results' and 'logs' subdirectories will be created here.")
    parser.add_argument("-c", "--compose-file", type=str, default="docker-compose.yml",
                        help="Path to the Docker Compose file for Joern servers.")
    parser.add_argument("-n", "--num-workers", type=int, default=1,
                        help="Number of parallel threads/workers (and Joern instances) to use.")
    parser.add_argument("--base-joern-port", type=int, default=16240,
                        help="Starting port number for Joern servers.")
    parser.add_argument("--joern-recreate-interval", type=int, default=5,
                        help="Number of samples to process before recreating a Joern server instance.")

    # LLM related arguments
    parser.add_argument("--llm-model-type", type=str, choices=["vLLM", "DeepSeek"], default="vLLM", # Default based on original code
                        help="Identifier string for the type of LLM model to use (e.g., 'vLLM', 'DeepSeek'). Passed to LLMManager.")
    parser.add_argument("--llm-model-name", "--llm-endpoint", dest="llm_model_name", type=str, required=True,
                        help="Model name passed to the API payload. '--llm-endpoint' is supported as a deprecated alias.")
    parser.add_argument("--llm-port", type=int, default=9001,
                        help="Port number for the LLM service.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if not os.path.isfile(args.dataset_file):
        logging.error(f"Dataset file not found: {args.dataset_file}")
        sys.exit(1)
    if args.num_workers <= 0:
        logging.error("--num-workers must be positive.")
        sys.exit(1)
    # Add more validation as needed

    # --- Setup Directories ---
    results_dir = os.path.join(args.output_base_dir, "results")
    logs_dir = os.path.join(args.output_base_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    logging.info(f"Outputs will be saved under: {args.output_base_dir}")
    logging.info(f"Number of workers: {args.num_workers}")


    # --- Load and Slice Dataset ---
    try:
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if not isinstance(dataset, list):
             raise TypeError("Dataset file does not contain a valid JSON list.")
        logging.info(f"Loaded dataset with {len(dataset)} samples from {args.dataset_file}")
    except (json.JSONDecodeError, TypeError, FileNotFoundError, OSError) as e:
        logging.error(f"Failed to load or parse dataset file {args.dataset_file}: {e}")
        sys.exit(1)


    if not dataset:
        logging.warning("Dataset is empty. Exiting.")
        sys.exit(0)


    dataset = dataset[:50]
    # Divide dataset into slices for each worker
    slice_size = len(dataset) // args.num_workers
    remainder = len(dataset) % args.num_workers
    threads = []
    start_idx = 0

    # --- Create and Start Threads ---
    logging.info("Starting worker threads...")
    for i in range(args.num_workers):
        # Distribute remainder items across the first 'remainder' threads
        current_slice_size = slice_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_slice_size

        if start_idx >= len(dataset):
            logging.warning(f"Worker {i+1} has no data assigned, reducing effective worker count.")
            continue

        dataset_slice = dataset[start_idx:end_idx]

        output_file = os.path.join(results_dir, f'thread_{i+1}_results.json')
        logs_file = os.path.join(logs_dir, f'thread_{i+1}_logs.json')
        port = args.base_joern_port + i

        # Create processor instance with all necessary arguments
        processor = DatasetProcessor(
            port=port,
            dataset_slice=dataset_slice,
            output_file=output_file,
            logs_file=logs_file,
            compose_file=args.compose_file,
            llm_model_type=args.llm_model_type,
            llm_model_name=args.llm_model_name,
            llm_port=args.llm_port,
            joern_recreate_interval=args.joern_recreate_interval
        )

        thread = threading.Thread(target=processor.process_dataset, name=f"Worker-{i+1}")
        thread.start()
        threads.append(thread)
        logging.info(f"Started Worker-{i+1} (Port {port}) processing {len(dataset_slice)} samples. Output: {output_file}, Logs: {logs_file}")

        start_idx = end_idx


    # --- Wait for Threads to Complete ---
    for thread in threads:
        thread.join()

    logging.info("All worker threads have completed processing.")


if __name__ == "__main__":
    main()
