"""
WirelessJEPA Setup Verification Script

Run this to verify all components are working correctly.
Usage: python test_setup.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def test_imports():
    """Test all module imports"""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    try:
        from config import WirelessJEPAConfig, get_default_config
        print("✓ config module")
    except Exception as e:
        print(f"✗ config module: {e}")
        return False
    
    try:
        from dataset import SignalPreprocessor, RML2016Dataset, WirelessJEPADataLoader
        print("✓ dataset module")
    except Exception as e:
        print(f"✗ dataset module: {e}")
        return False
    
    try:
        from mask import MaskGenerator, SparseConvolutionMask
        print("✓ mask module")
    except Exception as e:
        print(f"✗ mask module: {e}")
        return False
    
    try:
        from model import ShuffleNetV2Encoder, WirelessJEPA, create_model
        print("✓ model module")
    except Exception as e:
        print(f"✗ model module: {e}")
        return False
    
    try:
        from pretrain import JEPALoss, PretrainingTrainer
        print("✓ pretrain module")
    except Exception as e:
        print(f"✗ pretrain module: {e}")
        return False
    
    try:
        from eval import DownstreamEvaluator, LinearProber
        print("✓ eval module")
    except Exception as e:
        print(f"✗ eval module: {e}")
        return False
    
    try:
        from utils import count_parameters, get_device, set_reproducibility
        print("✓ utils module")
    except Exception as e:
        print(f"✗ utils module: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration system"""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)
    
    try:
        from config import WirelessJEPAConfig, get_default_config
        
        config = get_default_config()
        assert config is not None
        print("✓ Default configuration loaded")
        
        # Test nested configs
        assert config.data.batch_size == 64
        assert config.training.num_epochs == 100
        assert config.model.feature_dim == 256
        print("✓ Configuration parameters accessible")
        
        # Test directory creation
        config.output_dir = "./test_outputs"
        config.create_directories()
        assert Path("./test_outputs").exists()
        print("✓ Directory creation works")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_signals_and_preprocessing():
    """Test signal preprocessing"""
    print("\n" + "="*60)
    print("Testing Signal Preprocessing")
    print("="*60)
    
    try:
        from dataset import SignalPreprocessor
        import torch
        
        preprocessor = SignalPreprocessor()
        
        # Test complex to real conversion
        complex_signal = np.random.randn(4, 256) + 1j * np.random.randn(4, 256)
        real_signal = preprocessor.complex_to_real(complex_signal)
        assert real_signal.shape == (2, 4, 256), f"Expected (2, 4, 256), got {real_signal.shape}"
        print("✓ Complex to real conversion")
        
        # Test normalization
        normalized = preprocessor.normalize_unitmax(real_signal)
        assert np.max(np.abs(normalized)) <= 1.0
        print("✓ Unitmax normalization")
        
        # Test upsampling
        upsampled = preprocessor.upsample_antenna_time(real_signal)
        assert upsampled.shape == (2, 256, 256), f"Expected (2, 256, 256), got {upsampled.shape}"
        print("✓ Antenna-time upsampling")
        
        # Test full pipeline
        signal = torch.randn(4, 256, dtype=torch.complex64)
        preprocessed = preprocessor.preprocess(signal)
        assert preprocessed.shape == (2, 256, 256)
        print("✓ Full preprocessing pipeline")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masking():
    """Test masking strategies"""
    print("\n" + "="*60)
    print("Testing Masking Strategies")
    print("="*60)
    
    try:
        from mask import MaskGenerator
        
        mask_gen = MaskGenerator(height=256, width=256)
        
        # Test all mask types
        for mask_type in ["random", "antenna", "temporal", "multiblock"]:
            mask = mask_gen.generate_mask(mask_type, mask_ratio=0.75)
            assert mask.shape == (256, 256), f"Expected (256, 256), got {mask.shape}"
            assert mask.min() >= 0 and mask.max() <= 1
            coverage = (1 - mask.mean()).item()
            print(f"✓ {mask_type.capitalize()} masking ({coverage:.1%} masked)")
            
            # Test latent mask
            latent_mask = mask_gen.get_latent_mask(mask, factor=16)
            assert latent_mask.shape == (16, 16)
        
        return True
    except Exception as e:
        print(f"✗ Masking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation and forward pass"""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)
    
    try:
        from config import get_default_config
        from model import create_model
        
        config = get_default_config()
        model = create_model(config)
        
        from utils import count_parameters
        total, trainable = count_parameters(model)
        print(f"✓ Model created: {trainable/1e6:.2f}M trainable parameters")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, 2, 256, 256).to(device)
        x_teacher = torch.randn(batch_size, 2, 256, 256).to(device)
        mask = torch.randint(0, 2, (batch_size, 256, 256)).float().to(device)
        mask_latent = torch.randint(0, 2, (batch_size, 16, 16)).float().to(device)
        
        with torch.no_grad():
            output = model(x, x_teacher, mask, mask_latent)
        
        assert 'context_features' in output
        assert 'predicted_features' in output
        assert 'target_features' in output
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Context features: {output['context_features'].shape}")
        print(f"  Predicted features: {output['predicted_features'].shape}")
        print(f"  Target features: {output['target_features'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    data_path = Path("./data/RML2016.10a_dict.pkl")
    
    if not data_path.exists():
        print(f"⚠ Data file not found: {data_path}")
        print("  Skipping data loading test (expected for fresh setup)")
        return True
    
    try:
        from dataset import WirelessJEPADataLoader, create_dataloader
        
        # Test loader creation
        loader = WirelessJEPADataLoader(str(data_path), batch_size=32)
        print(f"✓ Data loader created")
        print(f"  Modulations: {len(loader.get_modulation_list())}")
        print(f"  SNR levels: {len(loader.get_snr_list())}")
        
        # Test get loaders
        loaders = loader.get_loaders(few_shot=None)
        assert 'train' in loaders and 'val' in loaders and 'test' in loaders
        print(f"✓ Train/Val/Test loaders created")
        
        # Test batch
        for signals, labels in loaders['train']:
            assert signals.shape[0] <= 32
            assert signals.shape == (len(labels), 2, 256, 256)
            print(f"✓ Batch shape correct: {signals.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device():
    """Test device detection"""
    print("\n" + "="*60)
    print("Testing Device Detection")
    print("="*60)
    
    try:
        cuda_available = torch.cuda.is_available()
        device = torch.device('cuda' if cuda_available else 'cpu')
        
        if cuda_available:
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print(f"✓ CPU mode (CUDA not available)")
        
        # Test tensor operations
        x = torch.randn(2, 64).to(device)
        y = torch.randn(2, 64).to(device)
        z = torch.mm(x, y.t())
        print(f"✓ Tensor operations work on {device}")
        
        return True
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# WirelessJEPA Setup Verification")
    print("#"*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Preprocessing", test_signals_and_preprocessing),
        ("Masking", test_masking),
        ("Device", test_device),
        ("Model", test_model_creation),
        ("Data Loading", test_data_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓✓✓ All tests passed! Setup is complete. ✓✓✓")
        print("\nNext steps:")
        print("  1. python main.py --mode demo        # Run demo")
        print("  2. python main.py --mode pretrain    # Train model")
        print("  3. python main.py --mode eval        # Evaluate model")
        return 0
    else:
        print(f"\n✗✗✗ {total - passed} test(s) failed. ✗✗✗")
        print("\nFor help, check QUICKSTART.md or see error details above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
