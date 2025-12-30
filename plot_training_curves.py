import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob

def load_tensorboard_logs(log_dir, config_name='Baseline'):
    """Load scalar data from tensorboard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    available_tags = ea.Tags()['scalars']
    print(f"Available tags in {log_dir}: {available_tags}")
    
    steps = None
    total_loss = None
    
    # Load l_pix (always present)
    if 'losses/l_pix' in available_tags:
        l_pix_events = ea.Scalars('losses/l_pix')
        steps = np.array([event.step for event in l_pix_events])
        l_pix_values = np.array([event.value for event in l_pix_events])
        total_loss = l_pix_values.copy()
        print(f"  Loaded losses/l_pix: {len(steps)} points")
    else:
        print(f"Could not find losses/l_pix in {log_dir}")
        return None, None
    
    # Add perceptual loss if present
    if 'losses/l_percep' in available_tags:
        l_percep_events = ea.Scalars('losses/l_percep')
        l_percep_values = np.array([event.value for event in l_percep_events])
        total_loss += l_percep_values
        print(f"  Added losses/l_percep")
    
    # Add MS-SSIM loss if present
    if 'losses/l_msssim' in available_tags:
        l_msssim_events = ea.Scalars('losses/l_msssim')
        l_msssim_values = np.array([event.value for event in l_msssim_events])
        total_loss += l_msssim_values
        print(f"  Added losses/l_msssim")
    
    return steps.tolist(), total_loss.tolist()

def plot_training_curves():
    """Plot training curves for all three configurations."""
    
    # Configuration paths
    configs = {
        'Baseline (L1)': 'tb_logger/RetinexFormer_LOL_v1',
        'L1 + Perceptual': 'tb_logger/RetinexFormer_LOL_v1_Perceptual',
        'L1 + MS-SSIM': 'tb_logger/RetinexFormer_LOL_v1_MSSSIM'
    }
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        'Baseline (L1)': '#1f77b4',      # Blue
        'L1 + Perceptual': '#ff7f0e',    # Orange
        'L1 + MS-SSIM': '#2ca02c'        # Green
    }
    
    line_styles = {
        'Baseline (L1)': '-',
        'L1 + Perceptual': '--',
        'L1 + MS-SSIM': '-.'
    }
    
    all_data = {}
    
    # Load data for each configuration
    for config_name, log_path in configs.items():
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} does not exist. Skipping {config_name}")
            continue
        
        # Find the event file in the log directory
        event_files = glob.glob(os.path.join(log_path, 'events.out.tfevents.*'))
        
        if not event_files:
            print(f"Warning: No event files found in {log_path}")
            continue
        
        # Use the most recent event file
        log_dir = log_path
        steps, values = load_tensorboard_logs(log_dir, config_name)
        
        if steps is not None and values is not None:
            all_data[config_name] = {'steps': steps, 'values': values}
            print(f"Loaded {len(steps)} data points for {config_name}")
        else:
            print(f"Failed to load data for {config_name}")
    
    # Plot the data
    if not all_data:
        print("Error: No data could be loaded. Please check your tensorboard log directories.")
        return
    
    for config_name, data in all_data.items():
        steps = np.array(data['steps'])
        values = np.array(data['values'])
        
        # Optional: Apply smoothing for better visualization
        window_size = 50
        if len(values) > window_size:
            smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps[window_size-1:]
        else:
            smoothed_values = values
            smoothed_steps = steps
        
        # Plot raw data with transparency
        ax.plot(steps, values, 
                color=colors[config_name], 
                alpha=0.2, 
                linewidth=0.5)
        
        # Plot smoothed curve
        ax.plot(smoothed_steps, smoothed_values,
                label=config_name,
                color=colors[config_name],
                linestyle=line_styles[config_name],
                linewidth=2.5)
    
    # Customize the plot
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss Curves Comparison\nRetinexFormer on LOL v1 Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Set y-axis to log scale if losses vary significantly
    # ax.set_yscale('log')
    
    # Add vertical lines for learning rate restarts (at 46K and 150K iterations)
    ax.axvline(x=46000, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, 
               label='LR Restart')
    ax.axvline(x=150000, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add text annotations for restarts
    ax.text(46000, ax.get_ylim()[1]*0.95, 'Restart 1', 
            rotation=90, verticalalignment='top', fontsize=10, alpha=0.7)
    ax.text(150000, ax.get_ylim()[1]*0.95, 'Restart 2', 
            rotation=90, verticalalignment='top', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'training_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF for LaTeX
    output_path_pdf = 'training_curves_comparison.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"PDF plot saved to: {output_path_pdf}")
    
    plt.show()

def main():
    """Main function to generate training curves plot."""
    print("=" * 60)
    print("Training Loss Curves Plotter")
    print("=" * 60)
    print("\nLoading tensorboard logs...")
    
    try:
        plot_training_curves()
        print("\n" + "=" * 60)
        print("Plot generation complete!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("1. Tensorboard is installed: pip install tensorboard")
        print("2. Matplotlib is installed: pip install matplotlib")
        print("3. The tb_logger directories exist and contain event files")

if __name__ == '__main__':
    main()
