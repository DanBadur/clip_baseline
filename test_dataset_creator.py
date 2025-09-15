#!/usr/bin/env python3
"""
Test script for the enhanced dataset_creator.py with word limiting
"""

import subprocess
import sys
import os

def test_word_limiting():
    """Test different word limits with the enhanced dataset creator"""
    
    # Test image path (you can modify this)
    test_image = r"C:\Users\danba\Documents\VPR datasets\SF_XL\small\train\37.75\@0543258.73@4178904.49@10@S@037.75643@-122.50891@lcc1lbG0KoUEWxDRn39hXg@@0@@@@201711@@.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please update the test_image path in this script")
        return
    
    print("🧪 Testing Enhanced Dataset Creator with Word Limiting")
    print("=" * 60)
    
    # Test different word limits
    word_limits = [10, 20, 30, 0]  # 0 means no limit
    
    for limit in word_limits:
        print(f"\n🔍 Testing with {limit if limit > 0 else 'no'} word limit:")
        print("-" * 40)
        
        try:
            # Run the dataset creator with different word limits
            cmd = [
                sys.executable, "dataset_creator.py",
                "--image_path", test_image,
                "--model", "gemini",
                "--max_words", str(limit)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"✅ Success with {limit} word limit")
                # Extract and show the response
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('Processing') and not line.startswith('Model:') and not line.startswith('Max words:') and not line.startswith('---') and not line.startswith('Inference time:') and not line.startswith('Processing completed!'):
                        print(f"Response: {line.strip()}")
            else:
                print(f"❌ Error with {limit} word limit:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout with {limit} word limit")
        except Exception as e:
            print(f"❌ Exception with {limit} word limit: {e}")
    
    print(f"\n🎯 Test completed!")
    print(f"\n💡 Usage examples:")
    print(f"  # Limit to 15 words")
    print(f"  python dataset_creator.py --image_path {test_image} --max_words 15")
    print(f"  # Use InternVL model with 25 word limit")
    print(f"  python dataset_creator.py --image_path {test_image} --model internvl --max_words 25")
    print(f"  # No word limit")
    print(f"  python dataset_creator.py --image_path {test_image} --max_words 0")

if __name__ == "__main__":
    test_word_limiting()
