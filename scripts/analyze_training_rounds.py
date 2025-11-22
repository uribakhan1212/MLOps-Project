#!/usr/bin/env python3
"""
Analyze per-round training metrics from fallback or detailed files
"""
import json
import os
import argparse
import glob
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional

def load_round_metrics_from_fallback(fallback_dir: str) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load round metrics directly from fallback directory"""
    try:
        if not fallback_dir:
            fallback_pattern = "fallback_metrics/*"
            fallback_dirs = glob.glob(fallback_pattern)
            if not fallback_dirs:
                print("❌ No fallback directories found")
                return None
            fallback_dir = max(fallback_dirs)
            print(f"🔍 Using most recent fallback: {fallback_dir}")
        
        round_metrics = {}
        round_files = glob.glob(os.path.join(fallback_dir, "round_*_metrics.json"))
        
        for round_file in sorted(round_files):
            with open(round_file, 'r') as f:
                data = json.load(f)
            
            round_data = data.get('metrics', {})
            round_num = round_data.get('round')
            
            if round_num is not None:
                round_metrics[round_num] = {
                    "val_loss": round_data.get('val_loss'),
                    "val_accuracy": round_data.get('val_accuracy'),
                    "val_auc": round_data.get('val_auc')
                }
        
        return round_metrics if round_metrics else None
        
    except Exception as e:
        print(f"❌ Failed to load round metrics: {e}")
        return None

def load_round_metrics_from_detailed(detailed_file: str) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load round metrics from detailed metrics file"""
    try:
        with open(detailed_file, 'r') as f:
            data = json.load(f)
        
        return data.get('round_metrics', {})
        
    except Exception as e:
        print(f"❌ Failed to load from detailed file: {e}")
        return None

def analyze_metrics(round_metrics: Dict[int, Dict[str, Any]]) -> None:
    """Analyze and display round metrics"""
    if not round_metrics:
        print("❌ No round metrics to analyze")
        return
    
    # Convert to DataFrame for easier analysis
    df_data = []
    for round_num, metrics in round_metrics.items():
        df_data.append({
            'round': round_num,
            'val_loss': metrics.get('val_loss'),
            'val_accuracy': metrics.get('val_accuracy'),
            'val_auc': metrics.get('val_auc')
        })
    
    df = pd.DataFrame(df_data).sort_values('round')
    
    print("\n📊 Training Progress Analysis")
    print("=" * 50)
    print(f"Total rounds: {len(df)}")
    print(f"Best accuracy: {df['val_accuracy'].max():.4f} (Round {df.loc[df['val_accuracy'].idxmax(), 'round']})")
    print(f"Best AUC: {df['val_auc'].max():.4f} (Round {df.loc[df['val_auc'].idxmax(), 'round']})")
    print(f"Lowest loss: {df['val_loss'].min():.4f} (Round {df.loc[df['val_loss'].idxmin(), 'round']})")
    
    # Check for improvement trends
    final_acc = df['val_accuracy'].iloc[-1]
    initial_acc = df['val_accuracy'].iloc[0]
    improvement = final_acc - initial_acc
    
    print(f"\nImprovement from Round 1 to Final:")
    print(f"  Accuracy: {initial_acc:.4f} → {final_acc:.4f} ({improvement:+.4f})")
    
    # Display round-by-round table
    print(f"\n📋 Round-by-Round Metrics:")
    print(df.to_string(index=False, float_format='%.4f'))

def plot_metrics(round_metrics: Dict[int, Dict[str, Any]], output_dir: str = "reports") -> None:
    """Create plots of training progress"""
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        rounds = sorted(round_metrics.keys())
        accuracies = [round_metrics[r]['val_accuracy'] for r in rounds]
        losses = [round_metrics[r]['val_loss'] for r in rounds]
        aucs = [round_metrics[r]['val_auc'] for r in rounds]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
        axes[0].set_title('Validation Accuracy')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        # AUC plot
        axes[2].plot(rounds, aucs, 'g-o', linewidth=2, markersize=6)
        axes[2].set_title('Validation AUC')
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('AUC')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training progress plot saved to: {plot_path}")
        
        plt.close()
        
    except ImportError:
        print("⚠️ matplotlib not available, skipping plots")
    except Exception as e:
        print(f"⚠️ Failed to create plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze per-round training metrics")
    parser.add_argument("--fallback-dir", help="Fallback directory path (auto-detect if not specified)")
    parser.add_argument("--detailed-file", help="Detailed metrics JSON file")
    parser.add_argument("--output-dir", default="reports", help="Output directory for plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    print("📊 Analyzing Training Round Metrics")
    print("=" * 50)
    
    round_metrics = None
    
    # Try to load from detailed file first
    if args.detailed_file and os.path.exists(args.detailed_file):
        print(f"📁 Loading from detailed file: {args.detailed_file}")
        round_metrics = load_round_metrics_from_detailed(args.detailed_file)
    
    # Fall back to fallback directory
    if not round_metrics:
        print("📁 Loading from fallback directory...")
        round_metrics = load_round_metrics_from_fallback(args.fallback_dir)
    
    if not round_metrics:
        print("❌ No round metrics found")
        exit(1)
    
    # Analyze metrics
    analyze_metrics(round_metrics)
    
    # Create plots
    if not args.no_plots:
        plot_metrics(round_metrics, args.output_dir)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()