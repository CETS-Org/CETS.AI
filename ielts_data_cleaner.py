import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

print("🚀 Starting IELTS Dataset Cleaning Pipeline...")
print("=" * 80)

class IELTSDatasetCleaner:
    """Class để clean và phân tích IELTS Reading Dataset"""
    
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Bước 1: Load dataset từ JSONL"""
        print("\n📂 STEP 1: LOADING DATA")
        print("-" * 80)
        
        if not os.path.exists(self.input_file):
            print(f"❌ ERROR: File '{self.input_file}' không tồn tại!")
            print("Vui lòng đảm bảo file ielts_reading_dataset.jsonl ở cùng thư mục.")
            return False
        
        try:
            data_list = []
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Skipping line {i+1} due to JSON error")
            
            self.df = pd.DataFrame(data_list)
            print(f"✅ Loaded {len(self.df)} records successfully")
            print(f"📊 Columns: {list(self.df.columns)}")
            print(f"📐 Dataset shape: {self.df.shape}")
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading data: {str(e)}")
            return False
    
    def extract_features(self):
        """Trích xuất features từ dữ liệu"""
        print("\n🔍 STEP 2: EXTRACTING FEATURES")
        print("-" * 80)
        
        # Extract từ prompt
        self.df['prompt_length'] = self.df['prompt'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        self.df['prompt_word_count'] = self.df['prompt'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Extract từ completion
        self.df['completion_length'] = self.df['completion'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        self.df['completion_word_count'] = self.df['completion'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Extract passage word count
        def extract_passage_words(completion):
            if pd.isna(completion):
                return 0
            try:
                passage = str(completion).split('Questions:')[0]
                passage = passage.replace('Reading Passage:', '').strip()
                return len(passage.split())
            except:
                return 0
        
        self.df['passage_word_count'] = self.df['completion'].apply(extract_passage_words)
        
        # Extract số câu hỏi
        def count_questions(completion):
            if pd.isna(completion):
                return 0
            try:
                return str(completion).count('Answer:')
            except:
                return 0
        
        self.df['num_questions'] = self.df['completion'].apply(count_questions)
        
        print("✅ Extracted features:")
        print(f"   - prompt_length, prompt_word_count")
        print(f"   - completion_length, completion_word_count")
        print(f"   - passage_word_count")
        print(f"   - num_questions")
        
    def detect_missing_values(self):
        """Phát hiện missing values"""
        print("\n🔎 STEP 3: DETECTING MISSING VALUES")
        print("-" * 80)
        
        missing_summary = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': [self.df[col].isna().sum() for col in self.df.columns],
            'Missing_Percentage': [f"{self.df[col].isna().sum() / len(self.df) * 100:.2f}%" for col in self.df.columns]
        })
        
        print(missing_summary.to_string(index=False))
        
        # Kiểm tra empty strings
        empty_prompts = (self.df['prompt'].astype(str).str.strip() == '').sum()
        empty_completions = (self.df['completion'].astype(str).str.strip() == '').sum()
        
        print(f"\n📝 Empty strings:")
        print(f"   - prompt: {empty_prompts}")
        print(f"   - completion: {empty_completions}")
        
    def handle_missing_values(self):
        """Xử lý missing values"""
        print("\n🧹 STEP 4: HANDLING MISSING VALUES")
        print("-" * 80)
        
        self.cleaned_df = self.df.copy()
        initial_count = len(self.cleaned_df)
        
        # Xóa rows có null values
        self.cleaned_df = self.cleaned_df.dropna(subset=['prompt', 'completion'])
        removed_null = initial_count - len(self.cleaned_df)
        print(f"✅ Removed {removed_null} rows with NULL values")
        
        # Xóa rows có empty strings
        self.cleaned_df = self.cleaned_df[
            (self.cleaned_df['prompt'].astype(str).str.strip() != '') & 
            (self.cleaned_df['completion'].astype(str).str.strip() != '')
        ]
        removed_empty = initial_count - removed_null - len(self.cleaned_df)
        print(f"✅ Removed {removed_empty} rows with empty strings")
        
        # Imputation cho numerical features
        numerical_cols = ['passage_word_count', 'num_questions']
        for col in numerical_cols:
            if col in self.cleaned_df.columns:
                median_val = self.cleaned_df[col].median()
                before = self.cleaned_df[col].isna().sum()
                self.cleaned_df[col].fillna(median_val, inplace=True)
                if before > 0:
                    print(f"✅ Imputed {before} missing values in {col} with median: {median_val:.0f}")
        
        print(f"\n📊 Dataset after handling missing values: {len(self.cleaned_df)} records")
        
    def handle_outliers(self):
        """Xử lý outliers sử dụng IQR method"""
        print("\n📉 STEP 5: DETECTING AND HANDLING OUTLIERS")
        print("-" * 80)
        
        initial_count = len(self.cleaned_df)
        
        # Kiểm tra passage_word_count (500-700 từ)
        print("\n1️⃣  Checking passage_word_count (expected: 500-700 words)")
        Q1 = self.cleaned_df['passage_word_count'].quantile(0.25)
        Q3 = self.cleaned_df['passage_word_count'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        print(f"   📊 Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
        print(f"   📏 IQR bounds: [{lower:.0f}, {upper:.0f}]")
        
        invalid_passage = (self.cleaned_df['passage_word_count'] < 500) | \
                         (self.cleaned_df['passage_word_count'] > 700)
        print(f"   ⚠️  Records outside 500-700 range: {invalid_passage.sum()}")
        
        # Kiểm tra num_questions (10-15 câu)
        print("\n2️⃣  Checking num_questions (expected: 10-15 questions)")
        Q1 = self.cleaned_df['num_questions'].quantile(0.25)
        Q3 = self.cleaned_df['num_questions'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        print(f"   📊 Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
        print(f"   📏 IQR bounds: [{lower:.0f}, {upper:.0f}]")
        
        invalid_questions = (self.cleaned_df['num_questions'] < 10) | \
                           (self.cleaned_df['num_questions'] > 15)
        print(f"   ⚠️  Records outside 10-15 range: {invalid_questions.sum()}")
        
        # Remove outliers
        self.cleaned_df = self.cleaned_df[
            (self.cleaned_df['passage_word_count'] >= 500) &
            (self.cleaned_df['passage_word_count'] <= 700) &
            (self.cleaned_df['num_questions'] >= 10) &
            (self.cleaned_df['num_questions'] <= 15)
        ]
        
        removed = initial_count - len(self.cleaned_df)
        print(f"\n✅ Removed {removed} outlier records")
        print(f"📊 Remaining records: {len(self.cleaned_df)}")
        
    def univariate_analysis(self):
        """Phân tích univariate"""
        print("\n📊 STEP 6: UNIVARIATE ANALYSIS")
        print("-" * 80)
        
        features = ['passage_word_count', 'num_questions']
        
        for feature in features:
            print(f"\n{'='*60}")
            print(f"📈 {feature.upper().replace('_', ' ')}")
            print(f"{'='*60}")
            
            # Central Tendency
            mean_val = self.cleaned_df[feature].mean()
            median_val = self.cleaned_df[feature].median()
            
            print(f"\n1. Central Tendency:")
            print(f"   Mean:   {mean_val:.2f}")
            print(f"   Median: {median_val:.2f}")
            
            # Dispersion
            std_val = self.cleaned_df[feature].std()
            min_val = self.cleaned_df[feature].min()
            max_val = self.cleaned_df[feature].max()
            
            print(f"\n2. Dispersion:")
            print(f"   Std Dev: {std_val:.2f}")
            print(f"   Min:     {min_val:.2f}")
            print(f"   Max:     {max_val:.2f}")
            
            # Skewness
            skewness = self.cleaned_df[feature].skew()
            print(f"\n3. Skewness: {skewness:.4f}")
            if abs(skewness) < 0.5:
                print("   ✅ Approximately symmetric")
            elif skewness > 0:
                print("   ⚠️  Positively skewed (right-skewed)")
            else:
                print("   ⚠️  Negatively skewed (left-skewed)")
            
            # Kurtosis
            kurtosis = self.cleaned_df[feature].kurtosis()
            print(f"\n4. Kurtosis: {kurtosis:.4f}")
            if abs(kurtosis) < 0.5:
                print("   ✅ Mesokurtic (normal distribution)")
            elif kurtosis > 0:
                print("   ⚠️  Leptokurtic (heavy-tailed)")
            else:
                print("   ⚠️  Platykurtic (light-tailed)")
    
    def create_visualizations(self):
        """Tạo biểu đồ"""
        print("\n📊 STEP 7: CREATING VISUALIZATIONS")
        print("-" * 80)
        
        try:
            # Histograms
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('Distribution Analysis - Histograms', fontsize=16, fontweight='bold')
            
            self.cleaned_df['passage_word_count'].hist(bins=30, ax=axes[0], color='skyblue', edgecolor='black')
            axes[0].set_title('Passage Word Count Distribution')
            axes[0].set_xlabel('Word Count')
            axes[0].set_ylabel('Frequency')
            axes[0].axvline(self.cleaned_df['passage_word_count'].mean(), color='red', linestyle='--', label='Mean')
            axes[0].legend()
            
            self.cleaned_df['num_questions'].hist(bins=15, ax=axes[1], color='lightgreen', edgecolor='black')
            axes[1].set_title('Number of Questions Distribution')
            axes[1].set_xlabel('Number of Questions')
            axes[1].set_ylabel('Frequency')
            axes[1].axvline(self.cleaned_df['num_questions'].mean(), color='red', linestyle='--', label='Mean')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: histograms.png")
            plt.close()
            
            # Boxplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('Outlier Detection - Boxplots', fontsize=16, fontweight='bold')
            
            self.cleaned_df.boxplot(column='passage_word_count', ax=axes[0])
            axes[0].set_title('Passage Word Count')
            axes[0].set_ylabel('Word Count')
            
            self.cleaned_df.boxplot(column='num_questions', ax=axes[1])
            axes[1].set_title('Number of Questions')
            axes[1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: boxplots.png")
            plt.close()
            
            # Scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(self.cleaned_df['passage_word_count'], 
                       self.cleaned_df['num_questions'], 
                       alpha=0.5, color='blue')
            plt.xlabel('Passage Word Count')
            plt.ylabel('Number of Questions')
            plt.title('Relationship: Passage Length vs Number of Questions')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: scatter_plot.png")
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create visualizations: {str(e)}")
    
    def generate_report(self):
        """Tạo báo cáo"""
        print("\n📋 STEP 8: DATA QUALITY REPORT")
        print("=" * 80)
        
        print(f"\n📊 Records Summary:")
        print(f"   Original:  {len(self.df):,}")
        print(f"   Cleaned:   {len(self.cleaned_df):,}")
        print(f"   Removed:   {len(self.df) - len(self.cleaned_df):,}")
        print(f"   Retention: {len(self.cleaned_df) / len(self.df) * 100:.2f}%")
        
        print(f"\n📈 Passage Word Count:")
        print(f"   Mean:   {self.cleaned_df['passage_word_count'].mean():.2f}")
        print(f"   Median: {self.cleaned_df['passage_word_count'].median():.2f}")
        print(f"   Range:  [{self.cleaned_df['passage_word_count'].min():.0f}, {self.cleaned_df['passage_word_count'].max():.0f}]")
        
        print(f"\n❓ Number of Questions:")
        print(f"   Mean:   {self.cleaned_df['num_questions'].mean():.2f}")
        print(f"   Median: {self.cleaned_df['num_questions'].median():.2f}")
        print(f"   Range:  [{self.cleaned_df['num_questions'].min():.0f}, {self.cleaned_df['num_questions'].max():.0f}]")
    
    def save_cleaned_data(self, output_file: str):
        """Lưu cleaned dataset"""
        print(f"\n💾 STEP 9: SAVING CLEANED DATASET")
        print("-" * 80)
        
        output_df = self.cleaned_df[['prompt', 'completion']].copy()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in output_df.iterrows():
                f.write(json.dumps({
                    'prompt': row['prompt'],
                    'completion': row['completion']
                }, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(output_df):,} records to: {output_file}")
        print(f"✅ Format: JSONL (ready for GPT-2 fine-tuning)")
    
    def run_pipeline(self, output_file: str):
        """Chạy toàn bộ pipeline"""
        if not self.load_data():
            return False
        
        self.extract_features()
        self.detect_missing_values()
        self.handle_missing_values()
        self.handle_outliers()
        self.univariate_analysis()
        self.create_visualizations()
        self.generate_report()
        self.save_cleaned_data(output_file)
        
        return True


# MAIN EXECUTION
if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("IELTS READING DATASET - CLEANING PIPELINE")
    print("🎯" * 40)
    
    # Tạo cleaner instance
    cleaner = IELTSDatasetCleaner("ielts_reading_dataset.jsonl")
    
    # Chạy pipeline
    success = cleaner.run_pipeline("ielts_reading_dataset_cleaned.jsonl")
    
    if success:
        print("\n" + "✅" * 40)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅" * 40)
        
        print("\n📁 Generated Files:")
        print("   1. ielts_reading_dataset_cleaned.jsonl - Cleaned dataset")
        print("   2. histograms.png - Distribution analysis")
        print("   3. boxplots.png - Outlier detection")
        print("   4. scatter_plot.png - Bivariate analysis")
        
        print("\n✨ Your dataset is ready for GPT-2 fine-tuning!")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")