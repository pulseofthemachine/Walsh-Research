import subprocess
import time
import sys
import re

def parse_candid_string(candid_out):
    # Candid output format: ("text content")
    # We need to extract the text content and unescape newlines/quotes
    match = re.search(r'^\("(.*)"\)$', candid_out, re.DOTALL)
    if match:
        content = match.group(1)
        # Unescape basic things
        content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        return content
    return candid_out

def benchmark(prompt, total_tokens, batch_size=5):
    current_text = prompt
    generated_count = 0
    start_time = time.time()
    
    print(f"Starting benchmark: {total_tokens} tokens, batch size {batch_size}")
    print(f"Initial prompt: {repr(prompt)}")
    
    while generated_count < total_tokens:
        # Determine tokens for this batch
        count = min(batch_size, total_tokens - generated_count)
        
        # Escape quotes for command line
        safe_prompt = current_text.replace('"', '\\"').replace('\n', '\\n')
        
        cmd = [
            "dfx", "canister", "call", "spinnet_backend", 
            "generate_full", 
            f'("{safe_prompt}", {count})'
        ]
        
        try:
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            # Parse output
            new_text = parse_candid_string(output)
            
            # Verify we got something new
            if len(new_text) <= len(current_text):
                print("Warning: Model returned same or shorter text. Stopping.")
                break
                
            # Update state
            added_text = new_text[len(current_text):]
            print(f"Generated ({count}): {repr(added_text)}")
            
            current_text = new_text
            generated_count += count
            
        except subprocess.CalledProcessError as e:
            print(f"Error calling canister: {e.stderr}")
            break
            
    end_time = time.time()
    duration = end_time - start_time
    tps = generated_count / duration if duration > 0 else 0
    
    print("-" * 40)
    print(f"Final Output:\n{current_text}")
    print("-" * 40)
    print(f"Summary:")
    print(f"Generated {generated_count} tokens in {duration:.2f}s")
    print(f"Speed: {tps:.2f} tok/s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark.py 'Prompt' [count]")
        sys.exit(1)
        
    prompt = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    benchmark(prompt, count)
