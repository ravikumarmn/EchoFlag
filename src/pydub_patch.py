"""
Patch for pydub to handle missing pyaudioop and aifc modules on Streamlit Cloud.
This provides fallback functionality when these modules are not available.
"""
import sys
import warnings

# Try to import aifc_patch
try:
    from aifc_patch import apply_patches as apply_aifc_patches
    apply_aifc_patches()
except ImportError:
    pass

# Try to patch before pydub is imported
def apply_patches():
    """Apply necessary patches to make pydub work without pyaudioop"""
    try:
        # Check if pyaudioop is already imported
        import pyaudioop
        return False  # No patching needed
    except ImportError:
        # Create a mock pyaudioop module
        class MockAudioop:
            def error(self):
                raise NotImplementedError("pyaudioop functionality not available")
                
            def __getattr__(self, name):
                warnings.warn(f"pyaudioop.{name} was called but is not available. Using fallback.")
                return self.error
        
        # Install the mock module
        sys.modules['pyaudioop'] = MockAudioop()
        
        # Patch pydub's audioop functions that we actually use
        try:
            from pydub import utils
            
            # Save original function
            original_get_min_max_value = getattr(utils, "get_min_max_value", None)
            
            # Create patched version
            def patched_get_min_max_value(bit_depth, signed=True):
                if original_get_min_max_value:
                    try:
                        return original_get_min_max_value(bit_depth, signed)
                    except Exception:
                        pass
                
                # Fallback implementation
                if signed:
                    min_value = -1 * (2 ** (bit_depth - 1))
                    max_value = (2 ** (bit_depth - 1)) - 1
                else:
                    min_value = 0
                    max_value = (2 ** bit_depth) - 1
                return min_value, max_value
            
            # Apply patch if utils is loaded
            if original_get_min_max_value:
                utils.get_min_max_value = patched_get_min_max_value
                
            return True  # Patching applied
            
        except ImportError:
            # pydub not imported yet, that's fine
            return True
