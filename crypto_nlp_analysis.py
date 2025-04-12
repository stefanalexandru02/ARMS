#!/usr/bin/env python3

import pandas as pd
import json
import requests
import time

def analyze_with_ollama(text, analysis_type, ollama_ip):
    """Send data to ollama for analysis"""
    url = f"http://{ollama_ip}:11434/api/generate"
    
    if analysis_type == "title":
        prompt = f"""
        Analyze this cryptocurrency news title: "{text}"
        
        Extract and return ONLY a JSON object with these keys:
        - verbs: list of all verbs in the title
        - adjectives: list of all adjectives in the title
        - tonality: string indicating if the tone is "positive", "negative", or "neutral"
        - sentiment_score: number between -1 and 1 indicating sentiment (-1 very negative, 1 very positive)
        
        Return only the JSON object, nothing else.
        """
    elif analysis_type == "text":
        prompt = f"""
        Analyze this cryptocurrency news text: "{text}"
        
        Extract and return ONLY a JSON object with these keys:
        - verbs: list of all verbs in the text
        - adjectives: list of all adjectives in the text
        - tonality: string indicating if the tone is "positive", "negative", or "neutral"
        - sentiment_score: number between -1 and 1 indicating sentiment (-1 very negative, 1 very positive)
        
        Return only the JSON object, nothing else.
        """
    else:  # combined analysis
        prompt = f"""
        Analyze this cryptocurrency news content:
        
        Title: "{text['title']}"
        
        Text: "{text['text']}"
        
        Extract and return ONLY a JSON object with these keys:
        - title_verbs: list of important verbs in the title
        - title_adjectives: list of important adjectives in the title
        - title_tonality: string indicating if the title tone is "positive", "negative", or "neutral"
        - text_verbs: list of important verbs in the text
        - text_adjectives: list of important adjectives in the text
        - text_tonality: string indicating if the text tone is "positive", "negative", or "neutral"
        - overall_sentiment: brief assessment (1-2 sentences) of the overall sentiment
        
        Return only the JSON object, nothing else.
        """
    
    data = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()["response"]
            # Try to extract just the JSON part from the response
            try:
                json_str = result.strip()
                # Find JSON beginning and end
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]
                    return json.loads(json_str)
                else:
                    return {"error": "No valid JSON found in response"}
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON from ollama response", "raw_response": result[:500]}
        return {"error": f"Ollama request failed with status {response.status_code}"}
    except Exception as e:
        return {"error": f"Exception when calling ollama: {str(e)}"}

def main():
    # Path to the CSV file
    csv_path = "data/cryptonews.csv"
    
    # Ollama IP address
    ollama_ip = "192.168.1.69"
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return
    
    # Create new columns to store the results
    result_columns = [
        'title_verbs', 'title_adjectives', 'title_tonality', 
        'text_verbs', 'text_adjectives', 'text_tonality',
        'combined_analysis'
    ]
    
    for col in result_columns:
        df[col] = None
    
    # Process each row in the dataframe
    total_rows = len(df)
    for idx, row in df.iterrows():
        print(f"Processing row {idx+1}/{total_rows}...")
        
        if not isinstance(row['title'], str) or not isinstance(row['text'], str):
            print(f"Skipping row {idx} - Invalid title or text format")
            continue
            
        try:
            # Process title (every row)
            title_result = analyze_with_ollama(row['title'], "title", ollama_ip)
            if "error" not in title_result:
                df.at[idx, 'title_verbs'] = json.dumps(title_result.get('verbs', []))
                df.at[idx, 'title_adjectives'] = json.dumps(title_result.get('adjectives', []))
                df.at[idx, 'title_tonality'] = title_result.get('tonality', 'unknown')
            else:
                print(f"Error analyzing title for row {idx}: {title_result.get('error')}")
            
            # Process text (every row)
            text_result = analyze_with_ollama(row['text'], "text", ollama_ip)
            if "error" not in text_result:
                df.at[idx, 'text_verbs'] = json.dumps(text_result.get('verbs', []))
                df.at[idx, 'text_adjectives'] = json.dumps(text_result.get('adjectives', []))
                df.at[idx, 'text_tonality'] = text_result.get('tonality', 'unknown')
            else:
                print(f"Error analyzing text for row {idx}: {text_result.get('error')}")
            
            # Process combined analysis (every 10th row to avoid overloading)
            if idx % 10 == 0:
                combined_result = analyze_with_ollama(
                    {'title': row['title'], 'text': row['text']}, 
                    "combined", 
                    ollama_ip
                )
                if "error" not in combined_result:
                    df.at[idx, 'combined_analysis'] = json.dumps(combined_result)
                else:
                    print(f"Error in combined analysis for row {idx}: {combined_result.get('error')}")
            
            # Add a short delay to prevent overwhelming the Ollama server
            time.sleep(1)
            
            # Save intermediate results every 50 rows
            if idx % 50 == 0 and idx > 0:
                intermediate_path = f"crypto_nlp_results_intermediate_{idx}.csv"
                df.to_csv(intermediate_path, index=False)
                print(f"Saved intermediate results to {intermediate_path}")
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    
    # Save the final results to CSV and JSON files
    output_path = "crypto_nlp_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Analysis complete. Results saved to {output_path}")
    
    # Also save a JSON version for easier processing
    json_output_path = "crypto_nlp_results.json"
    df.to_json(json_output_path, orient='records')
    print(f"JSON results saved to {json_output_path}")

if __name__ == "__main__":
    main()