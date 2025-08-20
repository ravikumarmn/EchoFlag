# Miscellaneous Documentation

## Version Compatibility Issues

### OpenAI + httpx Compatibility Fix (August 2025)

**Problem**: OpenAI library v1.37.0 has compatibility issues with httpx v0.28.1+, causing error:
```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

**Root Cause**: Newer httpx versions pass a `proxies` argument to OpenAI client initialization that isn't expected by OpenAI v1.37.0.

**Solution**: Pin httpx to v0.24.1 in requirements.txt to maintain compatibility with OpenAI v1.37.0.

**Fixed Files**:
- `requirements.txt`: Added `httpx==0.24.1`

**Installation Command**:
```bash
pip install httpx==0.24.1
```

**Verification**:
```python
from openai import OpenAI
client = OpenAI(api_key='test-key')  # Should work without errors
```

## Frequent Errors and Solutions

### Transcription Errors
- Always check OpenAI API key is set in environment
- Ensure httpx version compatibility (use v0.24.1)
- Verify audio file format is supported (mp3, mp4, wav)

### Installation Issues
- Use exact versions from requirements.txt
- If dependency conflicts arise, prioritize OpenAI and httpx compatibility
- Consider using virtual environment for clean installations
