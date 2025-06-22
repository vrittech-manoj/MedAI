#!/usr/bin/env python3
"""
Disease Analysis System using BioBERT
Console-based approach for processing disease data with NLP capabilities
Includes persistent storage for embeddings and model caching
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
import os
import pickle
import json
from datetime import datetime
import hashlib
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
    with persistent storage capabilities
    """
    
    def __init__(self, storage_dir="./disease_models"):
        print("🔬 Initializing BioBERT Disease Analyzer...")
        self.model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.storage_dir = storage_dir
        self.tokenizer = None
        self.model = None
        self.df = None
        self.embeddings = {}
        self.disease_profiles = []
        
        # Create storage directories
        self.create_storage_structure()
        
    def create_storage_structure(self):
        """Create directory structure for saving models and data"""
        directories = [
            self.storage_dir,
            os.path.join(self.storage_dir, "embeddings"),
            os.path.join(self.storage_dir, "models"),
            os.path.join(self.storage_dir, "datasets"),
            os.path.join(self.storage_dir, "predictions"),
            os.path.join(self.storage_dir, "logs")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"📁 Storage structure created at: {os.path.abspath(self.storage_dir)}")
        
    def get_data_hash(self, data_dict):
        """Generate hash for dataset to track changes"""
        data_str = json.dumps(data_dict, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
        
    def save_embeddings(self, data_hash):
        """Save generated embeddings to disk"""
        embeddings_file = os.path.join(
            self.storage_dir, "embeddings", f"disease_embeddings_{data_hash}.pkl"
        )
        
        try:
            # Prepare data for saving
            save_data = {
                'disease_profiles': self.disease_profiles,
                'data_hash': data_hash,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat(),
                'num_diseases': len(self.disease_profiles)
            }
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Save metadata
            metadata_file = os.path.join(
                self.storage_dir, "embeddings", f"metadata_{data_hash}.json"
            )
            metadata = {
                'data_hash': data_hash,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat(),
                'num_diseases': len(self.disease_profiles),
                'diseases': [p['disease'] for p in self.disease_profiles]
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"💾 Embeddings saved to: {embeddings_file}")
            print(f"📋 Metadata saved to: {metadata_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, data_hash):
        """Load previously generated embeddings from disk"""
        embeddings_file = os.path.join(
            self.storage_dir, "embeddings", f"disease_embeddings_{data_hash}.pkl"
        )
        
        if not os.path.exists(embeddings_file):
            return False
        
        try:
            with open(embeddings_file, 'rb') as f:
                save_data = pickle.load(f)
            
            self.disease_profiles = save_data['disease_profiles']
            print(f"✅ Loaded embeddings from: {embeddings_file}")
            print(f"📊 Loaded {len(self.disease_profiles)} disease profiles")
            print(f"🕒 Created: {save_data['created_at']}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
    
    def cache_model_locally(self):
        """Cache the BioBERT model locally"""
        model_cache_dir = os.path.join(self.storage_dir, "models", "biobert_cache")
        
        try:
            print("💾 Caching BioBERT model locally...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=model_cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                cache_dir=model_cache_dir
            )
            self.model.eval()
            
            # Save cache info
            cache_info = {
                'model_name': self.model_name,
                'cached_at': datetime.now().isoformat(),
                'cache_dir': model_cache_dir
            }
            
            with open(os.path.join(model_cache_dir, 'cache_info.json'), 'w') as f:
                json.dump(cache_info, f, indent=2)
                
            print(f"✅ BioBERT model cached at: {model_cache_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error caching model: {e}")
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
    
    def predict_disease_from_input(self, symptoms_input, causes_input=None, top_k=3):
        """Predict diseases based on user input symptoms and causes"""
        print(f"\n🔮 Predicting diseases for given symptoms...")
        print(f"Symptoms: {symptoms_input}")
        if causes_input:
            print(f"Causes: {causes_input}")
        
        if not hasattr(self, 'disease_profiles'):
            print("❌ Disease profiles not created. Run create_disease_profiles() first.")
            return
        
        # Create input profile
        if causes_input:
            input_text = f"Symptoms: {symptoms_input}. Causes: {causes_input}"
        else:
            input_text = f"Symptoms: {symptoms_input}"
        
        input_embedding = self.get_biobert_embedding(input_text)
        
        # Calculate similarities with all diseases
        predictions = []
        
        for profile in self.disease_profiles:
            # Compare with disease profile (symptoms + causes)
            similarity = cosine_similarity(
                [input_embedding], 
                [profile['full_embedding']]
            )[0][0]
            
            # Also compare just symptoms if available
            symptom_similarity = cosine_similarity(
                [self.get_biobert_embedding(symptoms_input)], 
                [profile['symptoms_embedding']]
            )[0][0]
            
            # Weighted average (more weight to full profile)
            combined_score = (similarity * 0.7) + (symptom_similarity * 0.3)
            
            predictions.append({
                'disease': profile['disease'],
                'confidence': combined_score,
                'symptoms_match': symptom_similarity,
                'full_match': similarity,
                'known_symptoms': profile['symptoms_text'],
                'known_causes': profile['causes_text']
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Display results and save log
        print(f"\n🎯 Top {min(top_k, len(predictions))} Disease Predictions:")
        print("=" * 60)
        
        for i, pred in enumerate(predictions[:top_k], 1):
            confidence_pct = pred['confidence'] * 100
            print(f"\n{i}. {pred['disease']} (Confidence: {confidence_pct:.1f}%)")
            print(f"   Known symptoms: {pred['known_symptoms']}")
            print(f"   Known causes: {pred['known_causes']}")
            print(f"   Symptom match: {pred['symptoms_match']:.3f}")
            print(f"   Overall match: {pred['full_match']:.3f}")
        
        # Save prediction log
        self.save_prediction_log(symptoms_input, causes_input, predictions[:top_k])
        
        return predictions[:top_k]
    
    def get_user_symptoms_and_causes(self):
        """Interactive symptom and cause input collection"""
        print("\n💬 Disease Prediction Assistant")
        print("=" * 40)
        
        # Get symptoms from user
        print("Please describe your symptoms:")
        print("(You can enter multiple symptoms separated by commas)")
        symptoms = input("🤒 Symptoms: ").strip()
        
        if not symptoms:
            print("❌ No symptoms provided!")
            return None, None
        
        # Get optional causes
        print("\nDo you know any possible causes? (Optional)")
        print("(Leave blank if unknown)")
        causes = input("🔬 Possible causes: ").strip()
        
        if not causes:
            causes = None
        
        return symptoms, causes
    
    def symptom_checker_mode(self):
        """Interactive symptom checker"""
        print("\n🏥 SYMPTOM CHECKER MODE")
        print("=" * 30)
        print("This tool helps predict possible diseases based on symptoms.")
        print("⚠️  DISCLAIMER: This is for educational purposes only.")
        print("   Always consult healthcare professionals for medical advice.")
        print("=" * 30)
        
        while True:
            try:
                print("\nOptions:")
                print("1. Enter new symptoms")
                print("2. Back to main menu")
                
                choice = input("\nChoose option (1-2): ").strip()
                
                if choice == '1':
                    symptoms, causes = self.get_user_symptoms_and_causes()
                    if symptoms:
                        predictions = self.predict_disease_from_input(symptoms, causes)
                        
                        # Ask if user wants more details
                        if predictions:
                            detail_choice = input("\n🔍 Want detailed analysis of top prediction? (y/n): ").strip().lower()
                            if detail_choice == 'y':
                                top_disease = predictions[0]['disease']
                                self.display_disease_details(top_disease)
                
                elif choice == '2':
                    break
                else:
                    print("❌ Invalid choice. Please enter 1 or 2.")
                    
            except KeyboardInterrupt:
                print("\n👋 Returning to main menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_disease_details(self, disease_name):
        """Display detailed information about a specific disease"""
        print(f"\n📋 Detailed Information: {disease_name}")
        print("=" * 40)
        
        # Find disease in original dataset
        disease_row = self.df[self.df['Disease'].str.lower() == disease_name.lower()]
        
        if disease_row.empty:
            print("❌ Disease details not found in dataset.")
            return
        
        row = disease_row.iloc[0]
        
        print(f"🦠 Disease: {row['Disease']}")
        print(f"🤒 Symptoms:")
        print(f"   • {row['Symptom1']}")
        print(f"   • {row['Symptom2']}")
        print(f"   • {row['Symptom3']}")
        print(f"🔬 Causes:")
        print(f"   • {row['Cause1']}")
        print(f"   • {row['Cause2']}")
        
        # Additional analysis
        print(f"\n📊 Quick Analysis:")
        similar = self.find_similar_diseases(disease_name, top_k=2)
        
    def interactive_query(self):
        """Enhanced interactive query system with disease prediction and file management"""
        print("\n🎮 Interactive Disease Analysis & Prediction System")
        print("=" * 50)
        print("Commands:")
        print("1. 'predict' - Predict diseases from symptoms")
        print("2. 'similar <disease>' - Find similar diseases")
        print("3. 'cluster' - Show disease clusters")
        print("4. 'symptoms' - Analyze symptom patterns")
        print("5. 'details <disease>' - Show disease details")
        print("6. 'files' - List saved files")
        print("7. 'regenerate' - Force regenerate embeddings")
        print("8. 'exit' - Exit system")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n🔬 Enter command: ").strip().lower()
                
                if query == 'exit':
                    print("👋 Goodbye!")
                    break
                elif query == 'predict':
                    self.symptom_checker_mode()
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
                elif query.startswith('details'):
                    parts = query.split(' ', 1)
                    if len(parts) > 1:
                        disease = parts[1].title()
                        self.display_disease_details(disease)
                    else:
                        print("❌ Please specify a disease. Example: 'details flu'")
                elif query == 'files':
                    self.list_saved_files()
                elif query == 'regenerate':
                    print("🔄 Regenerating embeddings...")
                    self.create_disease_profiles(force_regenerate=True)
                else:
                    print("❌ Unknown command. Try 'predict', 'similar <disease>', 'cluster', 'symptoms', 'details <disease>', 'files', 'regenerate', or 'exit'")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main execution function"""
    print("🚀 Starting BioBERT Disease Analysis & Prediction System")
    print("=" * 60)
    
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
    
    # Demo prediction with sample input
    print("\n🎯 Demo: Disease Prediction")
    print("Testing with sample symptoms...")
    sample_symptoms = "fever, cough, body ache"
    sample_causes = "virus, cold weather"
    analyzer.predict_disease_from_input(sample_symptoms, sample_causes, top_k=3)
    
    # Start interactive mode
    analyzer.interactive_query()

if __name__ == "__main__":
    main()