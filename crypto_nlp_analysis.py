#!/usr/bin/env python3

import pandas as pd
import json
import requests
import time
import os
import argparse
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm

# Constants
OLLAMA_IP = "192.168.1.69"
BATCH_SIZE = 10  # Process multiple records in a single prompt
MAX_WORKERS = 5  # Number of parallel workers
CHECKPOINT_INTERVAL = 100  # Save progress every N records
SAMPLE_RATE = 1.0  # 1.0 means analyze all records, 0.5 means analyze half, etc.

def create_batch_prompt(batch_data, analysis_type):
    """Create a prompt for batch processing"""
    if analysis_type == "title_text":
        items = []
        for i, item in enumerate(batch_data):
            items.append(f"RECORD {i+1}:\nTitle: \"{item['title']}\"\nText: \"{item['text'][:500]}\"")
        
        batch_text = "\n\n".join(items)
        
        prompt = f"""
        Analyze these {len(batch_data)} cryptocurrency news items:

        {batch_text}

        For EACH record, extract and return a JSON array where each element is an object with these keys:
        - record_id: the record number (1, 2, 3, etc.)
        - title_verbs: list of all verbs in the title
        - title_adjectives: list of all adjectives in the title
        - title_tonality: string indicating if the tone is "positive", "negative", or "neutral"
        - text_verbs: list of important verbs in the text (max 10)
        - text_adjectives: list of important adjectives in the text (max 10)
        - text_tonality: string indicating if the text tone is "positive", "negative", or "neutral"

        Return ONLY the JSON array, nothing else.
        """
        return prompt

def analyze_batch_with_ollama(batch_data, analysis_type):
    """Send batch data to ollama for analysis"""
    url = f"http://{OLLAMA_IP}:11434/api/generate"
    
    prompt = create_batch_prompt(batch_data, analysis_type)
    
    data = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()["response"]
                # Try to extract JSON from the response
                try:
                    json_str = result.strip()
                    # Find JSON beginning and end
                    start = json_str.find('[')
                    end = json_str.rfind(']') + 1
                    if start >= 0 and end > start:
                        json_str = json_str[start:end]
                        return json.loads(json_str)
                    else:
                        return {"error": "No valid JSON found in response"}
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON from ollama response", "raw_response": result[:500]}
            return {"error": f"Ollama request failed with status {response.status_code}"}
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                print(f"Request timed out. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return {"error": "Max retries reached due to timeout"}
        except Exception as e:
            return {"error": f"Exception when calling ollama: {str(e)}"}

def process_batch(batch_data, batch_idx):
    """Process a batch of records"""
    try:
        results = analyze_batch_with_ollama(batch_data, "title_text")
        
        if isinstance(results, list):
            return {
                "success": True,
                "batch_idx": batch_idx,
                "results": results
            }
        else:
            return {
                "success": False,
                "batch_idx": batch_idx,
                "error": results.get("error", "Unknown error")
            }
    except Exception as e:
        return {
            "success": False,
            "batch_idx": batch_idx,
            "error": str(e)
        }

def load_checkpoint(checkpoint_file):
    """Load progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"Resuming from checkpoint: {checkpoint_data['processed_records']} records processed")
            return checkpoint_data
        except:
            print("Error loading checkpoint. Starting from scratch.")
    
    return {
        "processed_records": 0,
        "processed_batches": [],
        "results": {}
    }

def save_checkpoint(checkpoint_file, checkpoint_data):
    """Save progress to checkpoint file"""
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {checkpoint_data['processed_records']} records processed")

def format_time(seconds):
    """Format time in seconds to a readable format"""
    return str(timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="Process cryptocurrency news data with Ollama")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--sample", type=float, default=SAMPLE_RATE, help="Sample rate (0.0-1.0)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    # Path to the CSV file
    csv_path = "/Users/stefanalexandru/Desktop/ARMS/data/cryptonews.csv"
    checkpoint_file = "/Users/stefanalexandru/Desktop/ARMS/crypto_nlp_checkpoint.json"
    output_path = "/Users/stefanalexandru/Desktop/ARMS/crypto_nlp_results.csv"
    json_output_path = "/Users/stefanalexandru/Desktop/ARMS/crypto_nlp_results.json"
    
    # Load checkpoint if resuming
    checkpoint_data = load_checkpoint(checkpoint_file) if args.resume else {
        "processed_records": 0,
        "processed_batches": [],
        "results": {}
    }
    
    # Use the command line arguments
    sample_rate = args.sample
    max_workers = args.workers
    batch_size = args.batch_size
    
    print(f"Using {max_workers} workers with batch size {batch_size} and sample rate {sample_rate}")
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        total_records = len(df)
        print(f"Loaded {total_records} rows from {csv_path}")
        
        # Apply sampling if specified
        if sample_rate < 1.0:
            sample_size = int(total_records * sample_rate)
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampling {sample_size} records ({sample_rate*100:.1f}% of total)")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return
    
    # Create result columns if they don't exist
    result_columns = [
        'title_verbs', 'title_adjectives', 'title_tonality', 
        'text_verbs', 'text_adjectives', 'text_tonality',
    ]
    
    for col in result_columns:
        if col not in df.columns:
            df[col] = None
    
    # Skip already processed batches
    start_idx = checkpoint_data["processed_records"]
    
    # Prepare batches
    batches = []
    batch_indices = []
    
    for i in range(start_idx, len(df), batch_size):
        if i + batch_size <= len(df):
            batch_data = []
            batch_idx = []
            for j in range(i, i + batch_size):
                if isinstance(df.iloc[j]['title'], str) and isinstance(df.iloc[j]['text'], str):
                    batch_data.append({
                        'title': df.iloc[j]['title'],
                        'text': df.iloc[j]['text']
                    })
                    batch_idx.append(j)
            
            if batch_data:
                batches.append(batch_data)
                batch_indices.append(batch_idx)
    
    total_batches = len(batches)
    print(f"Processing {total_batches} batches ({len(df) - start_idx} records remaining)")
    
    # Initialize progress bar
    progress_bar = tqdm(total=len(df) - start_idx)
    if start_idx > 0:
        progress_bar.update(0)  # Show progress from checkpoint
    
    start_time = time.time()
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, idx): (batch, i) 
            for i, (batch, idx) in enumerate(zip(batches, batch_indices))
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch, batch_num = future_to_batch[future]
            try:
                batch_result = future.result()
                
                if batch_result["success"]:
                    # Update dataframe with results
                    for result in batch_result["results"]:
                        record_id = result.get("record_id", 1) - 1  # Convert from 1-based to 0-based
                        if record_id < len(batch_indices[batch_num]):
                            idx = batch_indices[batch_num][record_id]
                            
                            df.at[idx, 'title_verbs'] = json.dumps(result.get('title_verbs', []))
                            df.at[idx, 'title_adjectives'] = json.dumps(result.get('title_adjectives', []))
                            df.at[idx, 'title_tonality'] = result.get('title_tonality', 'unknown')
                            df.at[idx, 'text_verbs'] = json.dumps(result.get('text_verbs', []))
                            df.at[idx, 'text_adjectives'] = json.dumps(result.get('text_adjectives', []))
                            df.at[idx, 'text_tonality'] = result.get('text_tonality', 'unknown')
                else:
                    print(f"\nError processing batch {batch_num}: {batch_result['error']}")
                
                # Update progress
                progress_bar.update(len(batch))
                
                # Calculate and display ETA
                elapsed_time = time.time() - start_time
                records_processed = min(progress_bar.n, len(df) - start_idx)
                if records_processed > 0:
                    seconds_per_record = elapsed_time / records_processed
                    eta_seconds = seconds_per_record * (len(df) - start_idx - records_processed)
                    progress_bar.set_description(f"ETA: {format_time(eta_seconds)}")
                
                # Update checkpoint
                checkpoint_data["processed_records"] = start_idx + progress_bar.n
                checkpoint_data["processed_batches"].append(batch_num)
                
                # Save checkpoint periodically
                if (start_idx + progress_bar.n) % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(checkpoint_file, checkpoint_data)
                    
                    # Also save intermediate results
                    intermediate_path = f"/Users/stefanalexandru/Desktop/ARMS/crypto_nlp_results_intermediate.csv"
                    df.to_csv(intermediate_path, index=False)
                    
            except Exception as e:
                print(f"\nError in batch {batch_num}: {str(e)}")
    
    progress_bar.close()
    
    # Save the final results to CSV and JSON files
    df.to_csv(output_path, index=False)
    print(f"\nAnalysis complete. Results saved to {output_path}")
    
    # Also save a JSON version for easier processing
    df.to_json(json_output_path, orient='records')
    print(f"JSON results saved to {json_output_path}")
    
    # Calculate and display statistics
    total_time = time.time() - start_time
    records_processed = len(df) - start_idx
    print(f"Processed {records_processed} records in {format_time(total_time)}")
    if records_processed > 0:
        print(f"Average: {total_time / records_processed:.2f} seconds per record")

if __name__ == "__main__":
    main()