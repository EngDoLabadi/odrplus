import json
import pandas as pd
import requests
import csv
import base64
import hashlib
import random
from typing import List, Dict, Any
from io import StringIO
from tqdm import tqdm
from collections import defaultdict

class BrowseCompStratifiedSampler:
    def __init__(self):
        """
        Initialize the sampler for BrowseComp questions using existing topic categories
        """
        self.target_proportions = {
            "TV shows & movies": 205,
            "Other": 197,
            "Science & technology": 173,
            "Art": 127,
            "History": 125,
            "Sports": 123,
            "Music": 116,
            "Video games": 71,
            "Geography": 70,
            "Politics": 59
        }
        
        # Calculate proportions for 60 questions
        total_original = sum(self.target_proportions.values())
        self.sample_proportions = {}
        for category, count in self.target_proportions.items():
            self.sample_proportions[category] = max(1, round((count / total_original) * 60))
        
        # Adjust to ensure we get exactly 60
        current_total = sum(self.sample_proportions.values())
        if current_total != 60:
            # Adjust the largest category
            largest_cat = max(self.sample_proportions.keys(), key=lambda k: self.sample_proportions[k])
            self.sample_proportions[largest_cat] += (60 - current_total)
    
    def derive_key(self, password, length):
        """Derive a fixed-length key from the password using SHA256."""
        hasher = hashlib.sha256()
        hasher.update(password.encode())
        key = hasher.digest()
        return key * (length // len(key)) + key[: length % len(key)]

    def decrypt(self, ciphertext_b64, password):
        """Decrypt base64-encoded ciphertext with XOR."""
        encrypted = base64.b64decode(ciphertext_b64)
        key = self.derive_key(password, len(encrypted))
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
        return decrypted.decode()
    
    def load_browsecomp_dataset(self, dataset_url: str) -> List[Dict[str, Any]]:
        """
        Load BrowseComp questions from the official dataset URL with decryption
        
        Args:
            dataset_url: URL to the BrowseComp CSV dataset
            
        Returns:
            List of questions with categories from the dataset
        """
        try:
            print(f"Downloading BrowseComp dataset from: {dataset_url}")
            df = pd.read_csv(dataset_url)
            
            print(f"Dataset loaded successfully!")
            print(f"   - Total rows: {len(df)}")
            print(f"   - Columns: {list(df.columns)}")
            
            questions_with_categories = []
            print("Decrypting questions...")
            
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
                try:
                    canary = row.get("canary", "")
                    problem = self.decrypt(row.get("problem", ""), canary)
                    answer = self.decrypt(row.get("answer", ""), canary) if pd.notna(row.get("answer", "")) else ""
                    problem_topic = row.get("problem_topic", "Other") 
                    
                    questions_with_categories.append({
                        "index": i,
                        "question": problem,
                        "answer": answer,
                        "category": problem_topic
                    })
                    
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    continue
            
            print(f"Successfully processed {len(questions_with_categories)} questions")
            
            category_counts = defaultdict(int)
            for q in questions_with_categories:
                category_counts[q['category']] += 1
            
            print(f"\nDataset Category Distribution:")
            for category in sorted(category_counts.keys()):
                count = category_counts[category]
                target = self.target_proportions.get(category, 0)
                print(f"   {category}: {count} questions (expected: {target})")
            
            return questions_with_categories
            
        except requests.RequestException as e:
            print(f"Error downloading dataset: {e}")
            raise
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV: {e}")
            raise
        except Exception as e:
            print(f" Unexpected error loading dataset: {e}")
            raise
    
    def create_stratified_sample(self, questions_with_categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a stratified random sample maintaining original proportions
        """
        # Filter out the first 60 questions (index 0-59) to avoid training set questions
        filtered_questions = [q for q in questions_with_categories if q['index'] > 59]
        
        # Group questions by category
        categorized_questions = defaultdict(list)
        for q in filtered_questions:
            categorized_questions[q['category']].append(q)
        
        print(f"\nCreating stratified sample of 60 questions...")
        print(f"Target distribution:")
        
        sample = []
        
        for category, target_count in self.sample_proportions.items():
            available_questions = categorized_questions.get(category, [])
            
            if len(available_questions) == 0:
                print(f"   {category}: {target_count} needed, 0 available - SKIPPING")
                continue
            
            # Sample randomly from this category
            if len(available_questions) >= target_count:
                sampled = random.sample(available_questions, target_count)
            else:
                sampled = available_questions
                print(f"   {category}: {target_count} needed, only {len(available_questions)} available")
            
            sample.extend(sampled)
            print(f"   {category}: {len(sampled)} questions selected")
        
        # If we still need more questions to reach 60, fill from "Other" or largest categories
        if len(sample) < 60:
            remaining_needed = 60 - len(sample)
            print(f"\n Need {remaining_needed} more questions to reach 60...")
            
            used_indices = set(q['index'] for q in sample)
            unused_questions = [q for q in filtered_questions if q['index'] not in used_indices]
            
            if len(unused_questions) >= remaining_needed:
                additional = random.sample(unused_questions, remaining_needed)
                sample.extend(additional)
                print(f"   Added {len(additional)} additional random questions")
        
        # Shuffle the final sample
        random.shuffle(sample)
        
        print(f"\nSample created: {len(sample)} questions total")
        
        return sample
    
    def save_sample(self, sample: List[Dict[str, Any]], base_filename: str = "browsecomp_randomized_sample_60"):
        """
        Save the stratified sample to multiple formats
        """
        
        # Add rank to each question
        for i, question in enumerate(sample):
            question['rank'] = i + 1
        
        # Save to CSV
        csv_filename = f"{base_filename}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['rank', 'index', 'question', 'answer', 'category']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in sample:
                writer.writerow(q)
        
        # Save to JSON
        json_filename = f"{base_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        
        # Save to TXT (human readable)
        txt_filename = f"{base_filename}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("BrowseComp: Stratified Random Sample (60 Questions)\n")
            f.write("Maintaining Original Category Proportions\n")
            f.write("=" * 70 + "\n\n")
            
            # Category summary
            category_counts = defaultdict(int)
            for q in sample:
                category_counts[q['category']] += 1
            
            f.write("Category Distribution:\n")
            for category, count in sorted(category_counts.items()):
                f.write(f"  {category}: {count} questions\n")
            f.write("\n" + "=" * 70 + "\n\n")
            
            # Questions
            for q in sample:
                f.write(f"Rank {q['rank']:2d} | Category: {q['category']}\n")
                f.write(f"Question: {q['question']}\n")
                f.write(f"Answer: {q['answer']}\n")
                f.write("-" * 70 + "\n\n")
        
        print(f"\nFiles created:")
        print(f"   {csv_filename} - CSV format")
        print(f"   {json_filename} - JSON format")  
        print(f"   {txt_filename} - Human readable")
        
        return csv_filename, json_filename, txt_filename


def create_stratified_browsecomp_sample():
    """
    Main function to create stratified random sample of BrowseComp questions
    """
    # Configuration
    DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    
    print("BrowseComp: Creating Stratified Random Sample (60 Questions)")
    print("Using existing problem_topic column for categorization")
    print("=" * 65)
    
    # Set random seed for reproducibility 
    random.seed(60)
    
    try:
        # Initialize sampler
        sampler = BrowseCompStratifiedSampler()
        
        print("Target sample distribution (60 questions total):")
        for category, count in sampler.sample_proportions.items():
            print(f"   {category}: {count} questions")
        
        questions_with_categories = sampler.load_browsecomp_dataset(DATASET_URL)
        
        sample = sampler.create_stratified_sample(questions_with_categories)
        
        csv_file, json_file, txt_file = sampler.save_sample(sample)
        
        # Print final summary
        category_counts = defaultdict(int)
        for q in sample:
            category_counts[q['category']] += 1
        
        print(f"\nFinal Sample Distribution:")
        for category in sorted(category_counts.keys()):
            target = sampler.sample_proportions.get(category, 0)
            actual = category_counts[category]
            percentage = (actual / 60) * 100
            print(f"   {category}: {actual}/{target} questions ({percentage:.1f}%)")
        
        print(f"\nStratified sample created successfully!")
        print(f"Use {csv_file} for your analysis")
        
        # Show a few sample questions from different categories
        print(f"\nSample questions from different categories:")
        shown_categories = set()
        for q in sample[:20]:  # Look at first 20 to find variety
            if q['category'] not in shown_categories and len(shown_categories) < 5:
                shown_categories.add(q['category'])
                print(f"\n[{q['category']}]")
                print(f"Q: {q['question'][:100]}...")
                print(f"A: {q['answer']}")
        
        return sample
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    create_stratified_browsecomp_sample()
