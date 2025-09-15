#!/usr/bin/env python3
"""
Test script to verify CLIP baseline setup
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__} imported successfully")
        
        import clip
        print("✓ CLIP imported successfully")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")
        
        from PIL import Image
        print("✓ PIL/Pillow imported successfully")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
        
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__} imported successfully")
        
        import tqdm
        print(f"✓ TQDM {tqdm.__version__} imported successfully")
        
        print("\n🎉 All packages imported successfully! CLIP baseline is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_clip_model():
    """Test if CLIP model can be loaded"""
    try:
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting CLIP model loading on {device}...")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✓ CLIP model loaded successfully")
        
        # Test text encoding
        text = "a photo of a cat"
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        print("✓ Text encoding works")
        
        # Test image encoding (dummy image)
        dummy_image = torch.randn(3, 224, 224).to(device)
        with torch.no_grad():
            image_features = model.encode_image(dummy_image.unsqueeze(0))
        print("✓ Image encoding works")
        
        print("✓ CLIP model is fully functional!")
        return True
        
    except Exception as e:
        print(f"❌ CLIP model test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CLIP Baseline Setup")
    print("=" * 40)
    
    imports_ok = test_imports()
    
    if imports_ok:
        test_clip_model()
    
    print("\nSetup test completed!")
