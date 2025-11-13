# Quick test to debug tokenizer
from filename_tokenizer import FilenameTokenizer, FilenamePatternAnalyzer

# Test with actual Visible Human filenames
test_files = [
    "VHMCT1mm-Hip (134).dcm",
    "VHMCT1mm-Hip (135).dcm", 
    "VHMCT1mm-Head (256).dcm",
    "VHFCT1mm-Hip (89).dcm",
    "VHFCT1mm-Head (120).dcm",
]

print("Testing tokenization:")
for fname in test_files[:3]:
    tokens = FilenameTokenizer.tokenize(fname)
    print(f"{fname}")
    print(f"  -> {tokens}")
    print()

print("\nTesting prefix extraction:")
analyzer = FilenamePatternAnalyzer(test_files)
stats = analyzer.analyze_token_statistics()

print("Prefix frequency:")
for prefix, count in stats['prefix_frequency'].items():
    print(f"  '{prefix}': {count}")

print("\nDominant prefixes:")
for p in stats['dominant_prefixes']:
    print(f"  '{p['prefix']}': {p['count']} files ({p['percentage']}%)")
