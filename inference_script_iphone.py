#!/usr/bin/env python3
"""
ML Model Inference and Post-Processing Script

This script loads a pretrained model, runs inference on exercise data,
performs segment analysis, and applies repetition counting algorithms.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle

# Import utility functions
from utils.utils import (
    load_pretrained_model,
    load_and_preprocess_data,
    create_dataloader,
    get_model_predictions_with_meta,
    find_segments,
    create_segment_dataframes,
    fuzzy_match_segments,
    ResearchBasedRepCounter,
    apply_research_counter_to_dataframe,
    IMUDatasetWithMeta,
    custom_collate_fn,
    process_custom_data
)
from utils.constants import rename_dict, name_fix_dict

# make a new file for inferenece for iphone

# add function arguments for model path, data path, download flag, save path, batch size, confidence threshold, min segment length, min rest length
def main(data_path='data/inference_test_data.csv',
         download=True,
         save_path='results.csv',
         batch_size=32,
         confidence_threshold=0.5,
         min_segment_length=7,
         min_rest_length=9,
         save_all_data = False):
    
    
    print("="*60)
    print("ML MODEL INFERENCE AND POST-PROCESSING")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist")
        sys.exit(1)
    
    try:

        
        # 2. Load and preprocess data
        print(f"\n1. Loading data from {data_path}")
        df = pd.read_pickle(data_path) if data_path.endswith('.pkl') else process_custom_data([data_path], clip=(0,0))

        print(f"   Loaded {len(df)} samples with {len(df['activity_name'].unique())} unique activities")
        

        # 5. Find segments
        print(f"\n2. Finding exercise sets")
        print("   Finding segments...")
        segments = find_segments(
            df, df['activity_name'], df['subject_id'], df.index, label_encoder=None, 
            use_predictions=False, 
            min_segment_length=min_segment_length, 
            min_rest_length=min_rest_length
        )

        
        print(f"   Found {len(segments)} segments")
        
        # 6. Create segment dataframes
        print(f"\n3. Creating segment dataframes")
        segments_df = create_segment_dataframes(segments, df, use_predictions=False)
    
        
        # 8. Apply repetition counting
        print(f"\n4. Applying research-based repetition counting")
        
        # Create repetition counter
        counter = ResearchBasedRepCounter(sampling_rate=50)

        
        # Apply counter to dataframe
        df_with_predictions = apply_research_counter_to_dataframe(
            segments_df, 
            signal_col='sig_array',
            exercise_col='exercise_name',
            counter=counter
        )
        
        print(f"   Generated repetition predictions for {len(df_with_predictions)} segments")
        
        # 9. Display results summary
        print(f"\n9. Results Summary")
        print(f"   Total segments processed: {len(df_with_predictions)}")
        
        if 'predicted_repetitions' in df_with_predictions.columns and 'repetitions' in df_with_predictions.columns:
            true_reps = df_with_predictions['repetitions']
            pred_reps = df_with_predictions['predicted_repetitions']
            
            mae = np.mean(np.abs(pred_reps - true_reps))
            accuracy_within_1 = np.mean(np.abs(pred_reps - true_reps) <= 1)
            accuracy_within_2 = np.mean(np.abs(pred_reps - true_reps) <= 2)
            
            print(f"   Mean Absolute Error: {mae:.2f} reps")
            print(f"   Accuracy (±1 rep): {accuracy_within_1:.1%}")
            print(f"   Accuracy (±2 reps): {accuracy_within_2:.1%}")
        
        # 10. Save results if requested
        if download:
            print(f"\n10. Saving results to {save_path}")

            if not save_all_data:
                # Remove signal array column if not needed
                df_with_predictions = df_with_predictions.drop(columns=['sig_array', 'indices', 'repetitions', 'confidence', 'exercise_label'], errors='ignore')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            # Save as CSV
            df_with_predictions.to_csv(save_path, index=False)
            print(f"    Results saved successfully to {save_path}")
            print(f"    Saved dataframe shape: {df_with_predictions.shape}")
        else:
            print(f"\n10. Results not saved (use --download flag to save)")
        
        print(f"\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return df_with_predictions
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()