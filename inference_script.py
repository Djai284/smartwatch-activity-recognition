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
def main():
    parser = argparse.ArgumentParser(description='Run ML model inference and post-processing')
    parser.add_argument('--model_path', type=str, default='runs/final_model_final_20250225_011001/',
                       help='Path to the model directory containing config and weights')
    parser.add_argument('--data_path', type=str, default = 'jacob_data.csv',
                       help='Path to the data file (.pkl or .csv)')
    parser.add_argument('--download', action='store_true',
                       help='Whether to save the resulting dataframe as CSV')
    parser.add_argument('--save_path', type=str, default='results.csv',
                       help='Path to save the CSV file if download is True')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for segment prediction')
    parser.add_argument('--min_segment_length', type=int, default=7,
                       help='Minimum segment length in data points')
    parser.add_argument('--min_rest_length', type=int, default=9,
                       help='Minimum consecutive rest periods to break segment')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ML MODEL INFERENCE AND POST-PROCESSING")
    print("="*60)
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        sys.exit(1)
    
    try:
        # 1. Load pretrained model
        print(f"\n1. Loading pretrained model from {args.model_path}")
        model, device = load_pretrained_model(args.model_path)
        model = model.to(device)
        print(f"   Model loaded successfully on device: {device}")
        
        # 2. Load and preprocess data
        print(f"\n2. Loading data from {args.data_path}")
        df = pd.read_pickle(args.data_path) if args.data_path.endswith('.pkl') else process_custom_data([args.data_path], clip=(0,0))
        
        # Filter unwanted activities
        # unwanted_activities = ['wallball', 'staticstretch', 'walk', 'repetitivestretching', 
        #                       'dynamicstretch(atyourownpace)', 'jumpingjacks']
        # df = df[~df['activity_name'].isin(unwanted_activities)]
       


        print(f"   Loaded {len(df)} samples with {len(df['activity_name'].unique())} unique activities")
        
        # 3. Prepare data for inference
        print(f"\n3. Preparing data for inference")
        


        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        # try to load a label_econder
        try:
            label_encoder_path = Path(args.model_path) / "label_encoder.pkl"
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
                label_classes = label_encoder.classes_
        except Exception as e:
            print(f"Error loading label encoder: {str(e)}")
            # label_encoder = None  
            # 
        df = df[df['activity_name'].isin(label_classes)]
        
        # Extract features and labels
        X_test = np.array(df['sig_array'])
        y_test = np.array(df['activity_name'])
        y_test_encoded = label_encoder.transform(y_test)

        
        # Convert to tensors
        X_test = np.stack(X_test).astype(np.float32)
        
        # Create dataset with metadata
        test_subject_ids = df['subject_id'].values
        test_original_indices = df.index.values
        
        test_dataset_with_meta = IMUDatasetWithMeta(X_test, y_test_encoded, test_subject_ids, test_original_indices)
        test_loader_with_meta = torch.utils.data.DataLoader(
            test_dataset_with_meta, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=custom_collate_fn
        )
        
        # 4. Run model predictions
        print(f"\n4. Running model inference")
        predictions, true_labels, subject_ids, original_indices, prediction_probs = get_model_predictions_with_meta(
            model, test_loader_with_meta, device
        )
        print(f"   Generated predictions for {len(predictions)} samples")
        
        # 5. Find segments
        print(f"\n5. Finding exercise segments")
        print("   Finding true segments...")
        true_segments = find_segments(
            df, true_labels, subject_ids, original_indices, label_encoder, 
            use_predictions=False, 
            min_segment_length=args.min_segment_length, 
            min_rest_length=args.min_rest_length
        )

        
        print("   Finding predicted segments...")
        pred_segments = find_segments(
            df, predictions, subject_ids, original_indices, label_encoder,
            confidence_threshold=args.confidence_threshold, 
            use_predictions=True, 
            prediction_probs=prediction_probs, 
            min_segment_length=args.min_segment_length, 
            min_rest_length=args.min_rest_length
        )
        
        print(f"   Found {len(true_segments)} true segments and {len(pred_segments)} predicted segments")
        
        # 6. Create segment dataframes
        print(f"\n6. Creating segment dataframes")
        true_segments_df = create_segment_dataframes(true_segments, df, label_encoder, use_predictions=False)
        pred_segments_df = create_segment_dataframes(pred_segments, df, label_encoder, use_predictions=True)
        
        # 7. Perform fuzzy matching
        print(f"\n7. Performing fuzzy segment matching")
        matches_df = fuzzy_match_segments(
            true_segments_df, pred_segments_df,
            min_overlap_ratio=0.3,
            min_iou=0.2,
            require_same_exercise=False,
            require_same_subject=True
        )
        
        print(f"   Found {len(matches_df)} segment matches")
        
        # 8. Apply repetition counting
        print(f"\n8. Applying research-based repetition counting")
        
        # Filter to segments with repetition data
        attempt = matches_df[matches_df['true_repetitions'] > 0]
        print(f"   Processing {len(attempt)} segments with repetition data")
        
        # Create repetition counter
        counter = ResearchBasedRepCounter(sampling_rate=50)
        
        # Apply counter to dataframe
        df_with_predictions = apply_research_counter_to_dataframe(
            attempt, 
            signal_col='true_sig_array',
            exercise_col='true_exercise_name',
            counter=counter
        )
        
        print(f"   Generated repetition predictions for {len(df_with_predictions)} segments")
        
        # 9. Display results summary
        print(f"\n9. Results Summary")
        print(f"   Total segments processed: {len(df_with_predictions)}")
        
        if 'predicted_repetitions' in df_with_predictions.columns and 'true_repetitions' in df_with_predictions.columns:
            true_reps = df_with_predictions['true_repetitions']
            pred_reps = df_with_predictions['predicted_repetitions']
            
            mae = np.mean(np.abs(pred_reps - true_reps))
            accuracy_within_1 = np.mean(np.abs(pred_reps - true_reps) <= 1)
            accuracy_within_2 = np.mean(np.abs(pred_reps - true_reps) <= 2)
            
            print(f"   Mean Absolute Error: {mae:.2f} reps")
            print(f"   Accuracy (±1 rep): {accuracy_within_1:.1%}")
            print(f"   Accuracy (±2 reps): {accuracy_within_2:.1%}")
        
        # 10. Save results if requested
        if args.download:
            print(f"\n10. Saving results to {args.save_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)

            
            
            # Save as CSV
            df_with_predictions.to_csv(args.save_path, index=False)
            print(f"    Results saved successfully to {args.save_path}")
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