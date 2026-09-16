"""
Unit tests for screen language detector module.
"""

from screen.language_detector import LanguageDetector

def test_detect_cpp_from_header():
    ocr_text = "LeetCode 4. Median of Two Sorted Arrays\nC++\nAuto\nclass Solution {\npublic:\n    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "C++"
    assert conf >= 0.9

def test_detect_cpp_from_syntax_only():
    ocr_text = "class Solution {\npublic:\n    vector<int> twoSum(vector<int>& nums, int target) {\n        unordered_map<int, int> seen;"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "C++"
    assert conf >= 0.85

def test_detect_java():
    ocr_text = "LeetCode 1. Two Sum\nJava\nAuto\nclass Solution {\n    public int[] twoSum(int[] nums, int target) {"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "Java"
    assert conf >= 0.9

def test_detect_python():
    ocr_text = "LeetCode 1. Two Sum\nPython3\nAuto\nclass Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "Python 3"
    assert conf >= 0.9

def test_detect_javascript():
    ocr_text = "LeetCode 1. Two Sum\nJavaScript\nAuto\nvar twoSum = function(nums, target) {"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "JavaScript"
    assert conf >= 0.9

def test_detect_typescript():
    ocr_text = "LeetCode 1. Two Sum\nTypeScript\nfunction twoSum(nums: number[], target: number): number[] {"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "TypeScript"
    assert conf >= 0.9

def test_detect_go():
    ocr_text = "LeetCode 1. Two Sum\nGo\nfunc twoSum(nums []int, target int) []int {"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "Go"
    assert conf >= 0.85

def test_detect_rust():
    ocr_text = "LeetCode 1. Two Sum\nRust\nimpl Solution {\n    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {"
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "Rust"
    assert conf >= 0.9

def test_detect_sql():
    ocr_text = "LeetCode 175. Combine Two Tables\nSQL Schema\nPandas\nWrite a solution to report the first name..."
    lang, conf = LanguageDetector.detect(ocr_text)
    assert lang == "SQL"
    assert conf >= 0.9
