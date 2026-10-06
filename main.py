from src.config_loader import load_raw_dataset
from src.encoding_domain import summarize_domains, encode_features
from src.information_theory import analyze_information_theory
from src.dimensionality import perform_pca_analysis
import pandas as pd  # <--- Import pandas
from src.encoded_data_export import save_encoded_dataset
from src.encoded_data_export import save_selected_dataset





def DataPreProcessing():
       print("==================================================================")
       print("   PHASE 2 - OBJECTIVE 1: MULTI-FACTOR DOMAIN CHARACTERIZATION   ")
       print("==================================================================")
   
       # 1. Load Data
       df_clean = load_raw_dataset()
   
       # 2. Domain Summary
       #summarize_domains(df_clean)
   
       # 3. Categorical Encoding & Dataset Splitting
       X, y, df_encoded = encode_features(df_clean)
       
       # 4. Save Encoded Data (Checks existence before writing)
       save_encoded_dataset(df_encoded)
   
       # 4. Information Theory Analysis
       info_df = analyze_information_theory(X, y)
   
       # 5. Dimensionality Reduction (PCA)
       pca_df = perform_pca_analysis(df_clean, X)
   
       # 6. Optimal Feature Selection
       optimal_features = info_df[info_df['Mutual_Information'] > 0.02]['Feature'].tolist()
       # 9. Save Selected Features Dataset
       save_selected_dataset(df_encoded, optimal_features)
       print("\n" + "="*50)
       print(f"✓ Objective 1 Execution Complete!")
       print(f"Selected {len(optimal_features)} optimal features (MI > 0.02):")
       for feat in optimal_features:
           print(f"  • {feat}")
       print("="*50)
    
    
    
    

def main():
    
# Below Function Perform following functions
#1-> Load Raw Dataset 
#2-> Encode the dataset 
#3-> Perfrom Information Theory
#4-> Dimensionality Reduction
#5-> Save Optimal Features into csv file in data folder 
 DataPreProcessing()
   

    
if __name__ == '__main__':
    main()