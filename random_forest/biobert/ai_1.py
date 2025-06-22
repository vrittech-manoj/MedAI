#!/usr/bin/env python3
"""
Disease Analysis System using BioBERT
Console-based approach for processing disease data with NLP capabilities
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Sample data
data = {
    "Disease": ["Flu", "Malaria", "Diabetes", "Asthma", "Migraine"],
    "Symptom1": ["Fever", "Fever", "Fatigue", "Shortness of breath", "Headache"],
    "Symptom2": ["Cough", "Chills", "Increased thirst", "Coughing", "Nausea"],
    "Symptom3": ["Body ache", "Sweating", "Frequent urination", "Wheezing", "Sensitivity to light"],
    "Cause1": ["Virus", "Parasite", "Insulin resistance", "Allergens", "Unknown"],
    "Cause2": ["Cold weather", "Mosquito bite", "Genetics", "Air pollution", "Stress"]
}

class BioBERTDiseaseAnalyzer:
    """
    Modern, scalable disease analysis system using BioBERT embeddings
    """
    
    def __init__(self):
        print("🔬 Initializing BioBERT Disease Analyzer...")
        self.model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.tokenizer = None
        self.model = None
        self.df = None
        self.embeddings = {}
        
    def load_model(self):
        """Load BioBERT model and tokenizer"""
        try:
            print("📥 Loading BioBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print("✅ BioBERT model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Using mock embeddings for demonstration...")
            return False
    
    def load_data(self, data_dict):
        """Load and preprocess disease data"""
        print("📊 Loading disease data...")
        self.df = pd.DataFrame(data_dict)
        print(f"✅ Loaded {len(self.df)} diseases")
        self.display_data_summary()
        
    def display_data_summary(self):
        """Display data summary"""
        print("\n" + "="*50)
        print("📋 DISEASE DATA SUMMARY")
        print("="*50)
        print(self.df.to_string(index=False))
        print(f"\nTotal diseases: {len(self.df)}")
        print(f"Columns: {list(self.df.columns)}")
        print("="*50)
    
    def get_biobert_embedding(self, text):
        """Generate BioBERT embedding for text"""
        if self.model is None:
            # Mock embedding for demonstration
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(768)
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", 
                                 truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"⚠️ Error generating embedding for '{text}': {e}")
            return np.random.randn(768)
    
    def create_disease_profiles(self):
        """Create comprehensive disease profiles with embeddings"""
        print("\n🧬 Creating disease profiles with BioBERT embeddings...")
        
        profiles = []
        
        for idx, row in self.df.iterrows():
            disease = row['Disease']
            print(f"Processing {disease}...")
            
            # Combine all text features
            symptoms_text = f"{row['Symptom1']}, {row['Symptom2']}, {row['Symptom3']}"
            causes_text = f"{row['Cause1']}, {row['Cause2']}"
            full_text = f"Disease: {disease}. Symptoms: {symptoms_text}. Causes: {causes_text}"
            
            # Generate embeddings
            disease_embedding = self.get_biobert_embedding(disease)
            symptoms_embedding = self.get_biobert_embedding(symptoms_text)
            causes_embedding = self.get_biobert_embedding(causes_text)
            full_embedding = self.get_biobert_embedding(full_text)
            
            profile = {
                'disease': disease,
                'symptoms_text': symptoms_text,
                'causes_text': causes_text,
                'full_text': full_text,
                'disease_embedding': disease_embedding,
                'symptoms_embedding': symptoms_embedding,
                'causes_embedding': causes_embedding,
                'full_embedding': full_embedding
            }
            
            profiles.append(profile)
        
        self.disease_profiles = profiles
        print("✅ Disease profiles created!")
        return profiles
    
    def find_similar_diseases(self, query_disease, top_k=3):
        """Find diseases similar to query disease"""
        print(f"\n🔍 Finding diseases similar to '{query_disease}'...")
        
        if not hasattr(self, 'disease_profiles'):
            print("❌ Disease profiles not created. Run create_disease_profiles() first.")
            return
        
        # Find query disease profile
        query_profile = None
        for profile in self.disease_profiles:
            if profile['disease'].lower() == query_disease.lower():
                query_profile = profile
                break
        
        if query_profile is None:
            print(f"❌ Disease '{query_disease}' not found in dataset.")
            return
        
        # Calculate similarities
        similarities = []
        query_embedding = query_profile['full_embedding']
        
        for profile in self.disease_profiles:
            if profile['disease'] != query_disease:
                similarity = cosine_similarity(
                    [query_embedding], 
                    [profile['full_embedding']]
                )[0][0]
                similarities.append((profile['disease'], similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Top {min(top_k, len(similarities))} similar diseases to {query_disease}:")
        print("-" * 40)
        for i, (disease, sim) in enumerate(similarities[:top_k], 1):
            print(f"{i}. {disease}: {sim:.3f}")
        
        return similarities[:top_k]
    
    def cluster_diseases(self, n_clusters=3):
        """Cluster diseases based on embeddings"""
        print(f"\n🎯 Clustering diseases into {n_clusters} groups...")
        
        if not hasattr(self, 'disease_profiles'):
            print("❌ Disease profiles not created. Run create_disease_profiles() first.")
            return
        
        # Prepare embeddings matrix
        embeddings_matrix = np.array([
            profile['full_embedding'] for profile in self.disease_profiles
        ])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_matrix)
        
        # Group diseases by cluster
        clusters = {}
        for i, profile in enumerate(self.disease_profiles):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(profile['disease'])
        
        print("\n📋 Disease Clusters:")
        print("=" * 30)
        for cluster_id, diseases in clusters.items():
            print(f"Cluster {cluster_id + 1}: {', '.join(diseases)}")
        
        return clusters
    
    def analyze_symptom_patterns(self):
        """Analyze symptom patterns across diseases"""
        print("\n💊 Analyzing symptom patterns...")
        
        # Collect all symptoms
        all_symptoms = []
        for idx, row in self.df.iterrows():
            symptoms = [row['Symptom1'], row['Symptom2'], row['Symptom3']]
            all_symptoms.extend(symptoms)
        
        # Count symptom frequencies
        from collections import Counter
        symptom_counts = Counter(all_symptoms)
        
        print("\n📊 Most common symptoms:")
        print("-" * 25)
        for symptom, count in symptom_counts.most_common():
            print(f"{symptom}: {count} disease(s)")
        
        return symptom_counts
    
    def interactive_query(self):
        """Interactive query system"""
        print("\n🎮 Interactive Disease Query System")
        print("=" * 40)
        print("Commands:")
        print("1. 'similar <disease>' - Find similar diseases")
        print("2. 'cluster' - Show disease clusters")
        print("3. 'symptoms' - Analyze symptom patterns")
        print("4. 'exit' - Exit system")
        print("=" * 40)
        
        while True:
            try:
                query = input("\n🔬 Enter command: ").strip().lower()
                
                if query == 'exit':
                    print("👋 Goodbye!")
                    break
                elif query.startswith('similar'):
                    parts = query.split(' ', 1)
                    if len(parts) > 1:
                        disease = parts[1].title()
                        self.find_similar_diseases(disease)
                    else:
                        print("❌ Please specify a disease. Example: 'similar flu'")
                elif query == 'cluster':
                    self.cluster_diseases()
                elif query == 'symptoms':
                    self.analyze_symptom_patterns()
                else:
                    print("❌ Unknown command. Try 'similar <disease>', 'cluster', 'symptoms', or 'exit'")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main execution function"""
    print("🚀 Starting BioBERT Disease Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BioBERTDiseaseAnalyzer()
    
    # Load model (will use mock if fails)
    model_loaded = analyzer.load_model()
    
    # Load data
    analyzer.load_data(data)
    
    # Create disease profiles
    analyzer.create_disease_profiles()
    
    # Run some initial analyses
    print("\n🔍 Running initial analyses...")
    analyzer.find_similar_diseases("Flu", top_k=2)
    analyzer.cluster_diseases(n_clusters=2)
    analyzer.analyze_symptom_patterns()
    
    # Start interactive mode
    analyzer.interactive_query()

if __name__ == "__main__":
    main()