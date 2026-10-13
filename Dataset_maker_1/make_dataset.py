# 01_dataset_maker/make_dataset.py
import json
import re
import requests
import uuid
import time
import config

TIMEOUT = 60 #seconds

def call_ollama(song_name, artist, description):
    """Sends context to Ollama, returns a list of prompts."""
    # Pass 'None' if description is empty so the LLM triggers the fallback instruction
    safe_desc = description.strip() if description else "None"
    user_content = f"Title: {song_name}\nArtist: {artist}\nDescription: {safe_desc}"
    
    payload = {
        "model": config.OLLAMA_MODEL,
        "system": config.SYSTEM_PROMPT,
        "prompt": user_content,
        "stream": False
    }
    
    try:
        # timeout prevents the script from hanging on a single song
        response = requests.post(config.OLLAMA_HOST, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        
        raw_text = response.json().get("response", "")
        prompts = [p.strip() for p in raw_text.strip().split('\n') if p.strip()]
        
        return prompts[:3]
    except Exception as e:
        print(f"  [!] Ollama API Error for '{song_name}': {e}")
        return []

def format_lyrics(raw_lyrics):
    """Converts [Verse 1] to <VERSE_1> and normalizes spacing."""
    formatted = re.sub(r'\[(.*?)\]', lambda m: f"<{m.group(1).upper().replace(' ', '_')}>", raw_lyrics)
    formatted = re.sub(r'\n+', ' ', formatted)
    return formatted.strip()

def process_dataset():
    # Telemetry tracking for run
    stats = {
        "total_attempted": 0,
        "success": 0,
        "failed": 0,
        "missing_description": 0,
        "missing_lyrics": 0
    }

    print(f"Starting overnight batch processing with {config.OLLAMA_MODEL}...\n")
    start_time = time.time()

    with open(config.INPUT_JSONL, 'r', encoding='utf-8') as infile, \
         open(config.OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            stats["total_attempted"] += 1
            
            try:
                data = json.loads(line)
                
                # Map to data
                song_name = data.get("song_name", "Unknown_Title")
                artist = data.get("artist", "Unknown_Artist")
                raw_lyrics = data.get("lyrics", "")
                description = data.get("description", "")
                
                # Generate a unique tracking ID for this song
                song_id = str(uuid.uuid4())
                
                if not raw_lyrics:
                    print(f"[{line_num}] SKIP: '{song_name}' (No lyrics found)")
                    stats["missing_lyrics"] += 1
                    stats["failed"] += 1
                    continue
                    
                if not description:
                    stats["missing_description"] += 1
                    
                print(f"[{line_num}] Processing: {song_name} by {artist} " + 
                      ("(Missing Desc -> Using Fallback)" if not description else ""))
                
                # Generate prompts via local LLM
                prompts = call_ollama(song_name, artist, description)
                
                if not prompts:
                    stats["failed"] += 1
                    continue
                    
                # Format the structural tags
                clean_lyrics = format_lyrics(raw_lyrics)
                
                # Save each prompt as its own training example, tied to the unique ID
                for prompt in prompts:
                    out_record = {
                        "song_id": song_id,
                        "song_name": song_name,
                        "artist": artist,
                        "prompt": prompt,
                        "lyrics": clean_lyrics
                    }
                    outfile.write(json.dumps(out_record) + '\n')
                    
                stats["success"] += 1
                
            except json.JSONDecodeError:
                print(f"[{line_num}] ERROR: Malformed JSON line.")
                stats["failed"] += 1
            except Exception as e:
                print(f"[{line_num}] UNEXPECTED ERROR on '{song_name}': {e}")
                stats["failed"] += 1

    # Final Telemetry Report
    elapsed = time.time() - start_time
    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Time Elapsed:         {elapsed / 60:.1f} minutes")
    print(f"Total Attempted:      {stats['total_attempted']}")
    print(f"Successful Songs:     {stats['success']}")
    print(f"Failed Songs:         {stats['failed']}")
    print(f"Missing Descriptions: {stats['missing_description']} (Handled via fallback)")
    print(f"Missing Lyrics:       {stats['missing_lyrics']} (Skipped)")
    print("="*40)

if __name__ == "__main__":
    process_dataset()