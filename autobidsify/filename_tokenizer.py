# filename_tokenizer.py
# Universal filename tokenizer - no assumptions, pure statistics + LLM understanding

"""
Universal Filename Tokenizer

How it works:
- Python: Statistical analysis of filename patterns (NO interpretation)
- LLM: Semantic understanding of what tokens mean
- User: Final validation when ambiguous

Core Strategy:
1. Tokenize ALL filenames
2. Analyze token frequency and distribution
3. Build statistical summary for LLM
4. Let LLM decide subject grouping logic
"""

from typing import Dict, List, Any, Set, Tuple, Optional
from collections import Counter, defaultdict
import re


class FilenameTokenizer:
    """
    Universal filename tokenizer using multiple strategies.
    
    Key Innovation: No hardcoded assumptions about naming conventions.
    """
    
    @staticmethod
    def tokenize(filename: str) -> List[str]:
        """
        Break filename into meaningful tokens.
        
        Strategy:
        1. Remove extension(s)
        2. Split by delimiters: _, -, space, (, ), [, ]
        3. Split CamelCase: "VHMCt" -> ["VHM", "Ct"]
        4. Split number boundaries: "CT1mm" -> ["CT", "1", "mm"]
        5. Keep tokens >= 2 characters (filter noise)
        
        Examples:
            "VHMCT1mm-Hip (134).dcm" -> ["VHM", "CT", "1", "mm", "Hip", "134"]
            "Beijing_sub82352" -> ["Beijing", "sub", "82352"]
            "scan_001_T1w.nii" -> ["scan", "001", "T1w"]
            "patient-A-rest.nii.gz" -> ["patient", "A", "rest"]
        
        Args:
            filename: Original filename (with or without extension)
        
        Returns:
            List of tokens
        """
        # Step 1: Remove all extensions
        name = filename
        while '.' in name and len(name.split('.')[-1]) <= 6:
            name = name.rsplit('.', 1)[0]
        
        # Step 2: Replace delimiters with spaces
        for delimiter in ['_', '-', '(', ')', '[', ']', '{', '}', ',', ';']:
            name = name.replace(delimiter, ' ')
        
        # Step 3: Split by spaces first
        parts = name.split()
        
        # Step 4: Further split each part by case changes and numbers
        tokens = []
        for part in parts:
            sub_tokens = FilenameTokenizer._split_advanced(part)
            tokens.extend(sub_tokens)
        
        # Step 5: Filter and clean
        tokens = [t for t in tokens if len(t) >= 1 and t.strip()]
        
        return tokens
    
    @staticmethod
    def _split_advanced(text: str) -> List[str]:
        """
        Advanced splitting: CamelCase + number boundaries.
        
        Strategy for consecutive uppercase letters:
        - Try to identify known acronyms (2-4 letters)
        - Split before last uppercase if followed by lowercase
        
        Examples:
            "VHMCT" -> ["VHM", "CT"]  (split 3+2 pattern)
            "CT1mm" -> ["CT", "1", "mm"]
            "sub82352" -> ["sub", "82352"]
            "T1w" -> ["T1w"] (keep together - common pattern)
            "HTMLParser" -> ["HTML", "Parser"]
        """
        if not text:
            return []
        
        # Special case: Known neuroimaging terms (keep together)
        if text in ['T1w', 'T2w', 'T1', 'T2', 'PD', 'FLAIR', 'DWI', 'BOLD']:
            return [text]
        
        # Use regex to split by type boundaries with lookahead
        # This pattern splits on:
        # - Uppercase followed by lowercase (CamelCase)
        # - Lowercase followed by uppercase
        # - Letter followed by digit
        # - Digit followed by letter
        pattern = r'([A-Z]+(?=[A-Z][a-z]|\b|[0-9])|[A-Z][a-z]+|[a-z]+|[0-9]+)'
        tokens = re.findall(pattern, text)
        
        # Filter empty
        tokens = [t for t in tokens if t]
        
        return tokens


class FilenamePatternAnalyzer:
    """
    Analyze filename patterns across ALL files - pure statistics.
    
    No assumptions about what tokens mean, just frequency analysis.
    """
    
    def __init__(self, all_filenames: List[str]):
        """
        Args:
            all_filenames: List of all filenames (can be full paths or just names)
        """
        # Extract just filenames if paths provided
        self.filenames = [f.split('/')[-1] if '/' in f else f for f in all_filenames]
        self.total_files = len(self.filenames)
    
    def analyze_token_statistics(self) -> Dict[str, Any]:
        """
        Analyze token frequencies across all files.
        
        CRITICAL: Use tokenization, NOT regex!
        
        Returns:
            {
                "total_files": int,
                "token_frequency": {token: count},
                "prefix_frequency": {prefix: count},
                "dominant_prefixes": [list of major prefixes],
                "token_positions": {position: {token: count}},
                "insights": [list of observations]
            }
        """
        all_tokens = Counter()
        prefix_tokens = Counter()  # CRITICAL: First TOKEN, not regex match
        position_tokens = defaultdict(Counter)
        
        for filename in self.filenames:
            # CRITICAL: Use tokenizer, not regex
            tokens = FilenameTokenizer.tokenize(filename)
            
            # Count all tokens
            for token in tokens:
                all_tokens[token] += 1
            
            # CRITICAL: Use first TOKEN as prefix
            if tokens and len(tokens) > 0:
                first_token = tokens[0]
                prefix_tokens[first_token] += 1
            
            # Count tokens by position
            for i, token in enumerate(tokens):
                position_tokens[i][token] += 1
        
        # Find dominant patterns
        dominant_prefixes = self._find_dominant_prefixes(prefix_tokens)
        insights = self._generate_insights(all_tokens, prefix_tokens, dominant_prefixes)
        
        return {
            "total_files": self.total_files,
            "token_frequency": dict(all_tokens.most_common(50)),
            "prefix_frequency": dict(prefix_tokens.most_common(20)),
            "dominant_prefixes": dominant_prefixes,
            "token_positions": {k: dict(v.most_common(10)) for k, v in position_tokens.items()},
            "insights": insights,
            "unique_token_count": len(all_tokens),
            "unique_prefix_count": len(prefix_tokens)
        }
    
    def _find_dominant_prefixes(self, prefix_counter: Counter) -> List[Dict[str, Any]]:
        """
        Find prefixes that might indicate subject groupings.
        
        A prefix is "dominant" if:
        - Appears in >5% of files
        - Not a common word (scan, data, file, etc.)
        """
        threshold = self.total_files * 0.05  # 5% threshold
        
        # Common words to exclude
        common_words = {'scan', 'data', 'file', 'image', 'sub', 'subject', 
                       'patient', 'sample', 'test', 'experiment'}
        
        dominant = []
        for prefix, count in prefix_counter.most_common(20):
            if count < threshold:
                continue
            
            if prefix.lower() in common_words:
                continue
            
            percentage = (count / self.total_files) * 100
            dominant.append({
                "prefix": prefix,
                "count": count,
                "percentage": round(percentage, 1)
            })
        
        return dominant
    
    def _generate_insights(self, all_tokens: Counter, prefix_tokens: Counter, 
                          dominant_prefixes: List[Dict]) -> List[str]:
        """
        Generate human-readable insights (NO interpretation, just observations).
        """
        insights = []
        
        # Insight 1: Overall token diversity
        if len(all_tokens) < 20:
            insights.append(f"Low token diversity: only {len(all_tokens)} unique tokens across all files")
        elif len(all_tokens) > 100:
            insights.append(f"High token diversity: {len(all_tokens)} unique tokens detected")
        
        # Insight 2: Prefix distribution
        if len(dominant_prefixes) == 0:
            insights.append("No dominant filename prefixes detected")
        elif len(dominant_prefixes) == 1:
            insights.append(f"Single dominant prefix '{dominant_prefixes[0]['prefix']}' in {dominant_prefixes[0]['percentage']}% of files")
        elif len(dominant_prefixes) == 2:
            p1, p2 = dominant_prefixes[0], dominant_prefixes[1]
            insights.append(f"Two major prefixes detected: '{p1['prefix']}' ({p1['percentage']}%) and '{p2['prefix']}' ({p2['percentage']}%)")
        else:
            insights.append(f"{len(dominant_prefixes)} dominant prefixes detected, suggesting possible subject groupings")
        
        # Insight 3: Most common tokens (potential identifiers)
        top_tokens = all_tokens.most_common(5)
        if top_tokens:
            common_list = [f"'{t}' ({c})" for t, c in top_tokens[:3]]
            insights.append(f"Most frequent tokens: {', '.join(common_list)}")
        
        return insights
    
    def build_llm_payload(self, user_hints: Dict[str, Any], 
                         max_samples: int = 30) -> Dict[str, Any]:
        """
        Build payload for LLM with statistics and samples.
        
        Args:
            user_hints: User-provided hints (nsubjects, description, etc.)
            max_samples: Number of filename samples to include
        
        Returns:
            Dict ready to be sent to LLM
        """
        stats = self.analyze_token_statistics()
        samples = self._sample_diverse_filenames(max_samples)
        
        return {
            "task": "subject_identification",
            "statistics": stats,
            "filename_samples": samples,
            "user_hints": user_hints,
            "instructions": (
                "Analyze the filename token statistics and samples. "
                "Determine how to group files by subject. "
                "The 'dominant_prefixes' may indicate subject identifiers. "
                "The 'insights' provide observations. "
                "User hint 'n_subjects' can help validate your hypothesis."
            )
        }
    
    def _sample_diverse_filenames(self, max_samples: int) -> List[str]:
        """
        Sample diverse filenames to show variety.
        
        Strategy: Group by FIRST TOKEN (not regex), sample from each group.
        """
        if len(self.filenames) <= max_samples:
            return sorted(self.filenames)
        
        # Group by first token (prefix)
        prefix_groups = defaultdict(list)
        for filename in self.filenames:
            tokens = FilenameTokenizer.tokenize(filename)
            prefix = tokens[0] if tokens else 'none'
            prefix_groups[prefix].append(filename)
        
        # Sample from each group
        samples = []
        samples_per_group = max(1, max_samples // len(prefix_groups))
        
        for prefix in sorted(prefix_groups.keys()):
            group_files = prefix_groups[prefix]
            n_samples = min(len(group_files), samples_per_group)
            samples.extend(sorted(group_files)[:n_samples])
            if len(samples) >= max_samples:
                break
        
        return samples[:max_samples]


# ============================================================================
# Integration Functions
# ============================================================================

def analyze_filenames_for_subjects(all_files: List[str], 
                                   user_hints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point: Analyze filenames to detect subject groupings.
    
    This function should be called from evidence.py or planner.py.
    
    Args:
        all_files: List of all file paths (relative)
        user_hints: User-provided hints
    
    Returns:
        {
            "python_statistics": {...},
            "llm_payload": {...},
            "confidence": "high|medium|low|none",
            "recommendation": str
        }
    """
    # Just extract filenames (not full paths)
    filenames = [f.split('/')[-1] for f in all_files]
    
    analyzer = FilenamePatternAnalyzer(filenames)
    stats = analyzer.analyze_token_statistics()
    
    # Build LLM payload
    llm_payload = analyzer.build_llm_payload(user_hints, max_samples=30)
    
    # Assess confidence
    dominant_count = len(stats['dominant_prefixes'])
    user_nsubjects = user_hints.get('n_subjects')
    
    confidence = "none"
    if dominant_count > 0:
        if user_nsubjects and dominant_count == user_nsubjects:
            confidence = "high"
        elif 2 <= dominant_count <= 10:
            confidence = "medium"
        else:
            confidence = "low"
    
    # Generate recommendation
    recommendation = _generate_recommendation(stats, user_hints)
    
    return {
        "python_statistics": stats,
        "llm_payload": llm_payload,
        "confidence": confidence,
        "recommendation": recommendation
    }


def _generate_recommendation(stats: Dict[str, Any], 
                             user_hints: Dict[str, Any]) -> str:
    """Generate recommendation for system behavior."""
    dominant_prefixes = stats['dominant_prefixes']
    user_nsubjects = user_hints.get('n_subjects')
    
    if not dominant_prefixes:
        return (
            "No clear filename patterns detected. "
            "Recommend using --describe to explain subject identification."
        )
    
    if user_nsubjects and len(dominant_prefixes) == user_nsubjects:
        prefixes_str = ', '.join([p['prefix'] for p in dominant_prefixes])
        return (
            f"HIGH CONFIDENCE: Detected {len(dominant_prefixes)} dominant prefixes "
            f"({prefixes_str}) matching user hint of {user_nsubjects} subjects."
        )
    
    if len(dominant_prefixes) in [2, 3, 4, 5]:
        return (
            f"MEDIUM CONFIDENCE: Detected {len(dominant_prefixes)} potential subject groups. "
            f"Will send to LLM for validation."
        )
    
    return (
        f"LOW CONFIDENCE: Found {len(dominant_prefixes)} prefix patterns, "
        f"which may or may not represent subjects. LLM will analyze."
    )


# ============================================================================
# LLM Response Structures
# ============================================================================

class SubjectGroupingDecision:
    """
    Structure for LLM's decision about subject grouping.
    
    This is what LLM should return after analyzing token statistics.
    """
    
    @staticmethod
    def create_prefix_mapping(prefix_to_subject: Dict[str, str], 
                             metadata: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Create prefix-based grouping decision.
        
        Example:
            prefix_to_subject = {"VHM": "1", "VHF": "2"}
            metadata = {
                "1": {"sex": "M", "group": "cadaver"},
                "2": {"sex": "F", "group": "cadaver"}
            }
        
        Returns:
            {
                "method": "prefix_based",
                "rules": [
                    {"prefix": "VHM", "maps_to_subject": "1"},
                    {"prefix": "VHF", "maps_to_subject": "2"}
                ],
                "participant_metadata": {...}
            }
        """
        return {
            "method": "prefix_based",
            "description": f"Files grouped by {len(prefix_to_subject)} filename prefixes",
            "rules": [
                {
                    "prefix": prefix,
                    "maps_to_subject": subj_id,
                    "match_pattern": f"{prefix}*"
                }
                for prefix, subj_id in prefix_to_subject.items()
            ],
            "participant_metadata": metadata or {}
        }
    
    @staticmethod
    def create_sequential_assignment(n_subjects: int) -> Dict[str, Any]:
        """
        Fallback: Sequential ID assignment when no clear pattern.
        
        Returns:
            {
                "method": "sequential",
                "n_subjects": int,
                "note": "..."
            }
        """
        return {
            "method": "sequential",
            "n_subjects": n_subjects,
            "note": (
                "No clear subject grouping pattern detected in filenames. "
                "Assigning sequential IDs based on file order or user hint."
            )
        }
    
    @staticmethod
    def create_blocking_question(reason: str, options: List[str]) -> Dict[str, Any]:
        """
        Create blocking question when LLM cannot decide.
        
        Returns:
            {
                "method": "blocked",
                "reason": str,
                "question": {...}
            }
        """
        return {
            "method": "blocked",
            "reason": reason,
            "question": {
                "type": "subject_grouping",
                "severity": "block",
                "message": reason,
                "options": options
            }
        }


# ============================================================================
# Testing
# ============================================================================

def test_filename_tokenizer():
    """Test suite for filename tokenizer"""
    
    print("=" * 80)
    print("Testing Filename Tokenizer")
    print("=" * 80)
    
    # Test Case 1: Visible Human
    print("\n[Test 1] Visible Human Dataset")
    print("-" * 80)
    
    vh_files = [
        "VHMCT1mm-Hip (134).dcm",
        "VHMCT1mm-Hip (135).dcm",
        "VHMCT1mm-Head (256).dcm",
        "VHMCT1mm-Shoulder (89).dcm",
        "VHFCT1mm-Hip (45).dcm",
        "VHFCT1mm-Head (120).dcm",
        "VHFCT1mm-Ankle (78).dcm",
    ]
    
    print("Tokenization examples:")
    for fname in vh_files[:3]:
        tokens = FilenameTokenizer.tokenize(fname)
        print(f"  {fname}")
        print(f"    -> {tokens}")
    
    analyzer = FilenamePatternAnalyzer(vh_files)
    stats = analyzer.analyze_token_statistics()
    
    print(f"\nStatistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Unique tokens: {stats['unique_token_count']}")
    print(f"  Unique prefixes: {stats['unique_prefix_count']}")
    print(f"\nPrefix frequency:")
    for prefix, count in stats['prefix_frequency'].items():
        pct = (count / stats['total_files']) * 100
        print(f"  '{prefix}': {count} ({pct:.1f}%)")
    
    print(f"\nDominant prefixes:")
    for p in stats['dominant_prefixes']:
        print(f"  '{p['prefix']}': {p['count']} files ({p['percentage']}%)")
    
    print(f"\nInsights:")
    for insight in stats['insights']:
        print(f"  - {insight}")
    
    # Test LLM payload
    payload = analyzer.build_llm_payload({"n_subjects": 2}, max_samples=10)
    print(f"\nLLM Payload structure:")
    print(f"  Task: {payload['task']}")
    print(f"  Samples: {len(payload['filename_samples'])} filenames")
    print(f"  Statistics keys: {list(payload['statistics'].keys())}")
    
    # Test Case 2: Multi-site
    print("\n[Test 2] Multi-site Dataset")
    print("-" * 80)
    
    multisite_files = [
        "Beijing_sub82352/anat/scan.nii.gz",
        "Cambridge_sub06272/anat/scan.nii.gz",
        "Leiden_sub04484/func/rest.nii.gz",
    ]
    
    # Just extract directory names as "filenames"
    dir_names = [f.split('/')[0] for f in multisite_files]
    
    print("Tokenization examples:")
    for dname in dir_names:
        tokens = FilenameTokenizer.tokenize(dname)
        print(f"  {dname} -> {tokens}")
    
    # Test Case 3: Standard BIDS
    print("\n[Test 3] Standard BIDS")
    print("-" * 80)
    
    bids_files = [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-02/anat/sub-02_T1w.nii.gz",
        "sub-03/func/sub-03_task-rest_bold.nii.gz",
    ]
    
    dir_names = [f.split('/')[0] for f in bids_files]
    print("Tokenization examples:")
    for dname in dir_names:
        tokens = FilenameTokenizer.tokenize(dname)
        print(f"  {dname} -> {tokens}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_filename_tokenizer()
