# Whisper versions

## Version 1 (v1)

### Removed Segments
- Segments by speaker1 (Annette Gerstenberg)
- Segments without text
- Segments shorter than 100 ms
- Segments below intensity threshold of 1e-6 ('silent segments')
- Segments containing text "(buzz) anon (buzz)"

### Train-Test Split
- Train: 80%
- Test: 20%
- No segments ascribed to test set via manual selection