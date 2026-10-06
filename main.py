import os
import pandas as pd
from src.config_loader import load_raw_dataset, DATA_DIR, TARGET_COLUMN
from src.encoding_domain import summarize_domains, encode_features
from src.information_theory import analyze_information_theory
from src.dimensionality import perform_pca_analysis
from src.encoded_data_export import save_encoded_dataset, save_selected_dataset
from src.datapreprocessed import run_preprocessing_pipeline, load_selected_dataset, remove_missing_rows, remove_duplicate_rows
from src.model_trainer import initialize_baseline_models, perform_cross_validation, train_models
from src.model_evaluator import evaluate_models, plot_confusion_matrices
from src.export_results import save_baseline_results




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
       #print("\n" + "="*50)
       #print(f"✓ Objective 1 Execution Complete!")
       #print(f"Selected {len(optimal_features)} optimal features (MI > 0.02):")
       #for feat in optimal_features:
           #print(f"  • {feat}")
       #print("="*50)
    
       
    
    

def main():
    
    # Below Function Perform following functions
    #1-> Load Raw Dataset 
    #2-> Encode the dataset 
    #3-> Perfrom Information Theory
    #4-> Dimensionality Reduction
    #5-> Save Optimal Features into csv file in data folder 
    #DataPreProcessing()   
    # ==================================================================
    #   PHASE 2 - OBJECTIVE 2: DATA PREPROCESSING PIPELINE
    # # ==================================================================


    #Single call to run missing/duplicate checks, export DataProcessed.csv, split, and scale features
    # 1. Single call to run missing/duplicate checks, export DataProcessed.csv, split, and scale features

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = run_preprocessing_pipeline()

    print("\n[Preprocessing Check] Ready for baseline model training!")
    print(f"  • X_train_scaled Shape: {X_train_scaled.shape}")
    print(f"  • X_test_scaled Shape:  {X_test_scaled.shape}")

    # 2. Load cleaned full unscaled dataset for Stratified 10-Fold CV using Pipeline
    df_selected = load_selected_dataset()
    df_clean_full = remove_duplicate_rows(remove_missing_rows(df_selected))
    X_full = df_clean_full.drop(columns=[TARGET_COLUMN])
    y_full = df_clean_full[TARGET_COLUMN]

    # 3. Initialize Baseline Models
    models = initialize_baseline_models()

    # 4. Perform Stratified 10-Fold Cross-Validation (Evaluates generalisation)
    cv_results_df = perform_cross_validation(X_full, y_full, models, n_splits=10)

    # 5. Train Models on Training Split and Evaluate on Holdout Test Split
    trained_models = train_models(models, X_train_scaled, y_train)
    test_metrics_df, detailed_reports = evaluate_models(trained_models, X_test_scaled, y_test)

    # 6. Export Benchmarks and Render Confusion Matrices
    save_baseline_results(cv_results_df, test_metrics_df, detailed_reports)

    cm_figure_path = os.path.join(DATA_DIR, "results", "confusion_matrices.png")
    plot_confusion_matrices(detailed_reports, save_path=cm_figure_path)

    print("\n==================================================================")
    print("✓ PHASE 2 OBJECTIVE 2 COMPLETE! BASELINE BENCHMARKS PROCESSED.")
    print("==================================================================")
    print("\nStratified 10-Fold Cross-Validation Summary:")
    print(cv_results_df.to_string(index=False)) 
if __name__ == '__main__':
    main()