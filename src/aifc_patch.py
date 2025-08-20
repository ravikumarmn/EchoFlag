"""
Patch for missing aifc module in Streamlit Cloud environment.
This provides a minimal mock implementation to prevent import errors.
"""
import sys
import warnings

def apply_patches():
    """Apply necessary patches to handle missing aifc module"""
    try:
        # Check if aifc is already imported
        import aifc
        return False  # No patching needed
    except ImportError:
        # Create a mock aifc module
        class MockAifc:
            def __init__(self):
                self.Error = Exception
                
            def error(self, *args, **kwargs):
                warnings.warn("aifc module function was called but is not available")
                raise NotImplementedError("aifc functionality not available")
            
            def __getattr__(self, name):
                warnings.warn(f"aifc.{name} was called but is not available")
                return self.error
        
        # Install the mock module
        sys.modules['aifc'] = MockAifc()
        warnings.warn("Installed mock aifc module due to import failure")
        return True
