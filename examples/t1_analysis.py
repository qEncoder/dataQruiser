import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import json
import base64
from io import BytesIO
from qdrive.dataset.search import search_datasets
from pathlib import Path
from qdrive import dataset
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.palettes import Category10

def exp_decay(t, A, T1, offset):
    return A * np.exp(-t / T1) + offset

def fit_t1(time, signal):
    try:
        popt, _ = curve_fit(exp_decay, time, signal, 
                            p0=[np.max(signal), time[len(time)//2], np.min(signal)])
        return popt[1]  # Return T1
    except:
        return np.nan
    



def t1_correlation_analysis(datasets, output_file='data/t1_correlation_report.html'):
    """
    Analyze multiple T1 datasets and correlate T1 values with parameters.
    
    Args:
        datasets: List of dataset objects to analyze
        output_file: Path to save the HTML report
    """
    
    def plot_to_b64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{img_b64}"
    
    all_results = []
    all_traces = []  # Store all T1 traces for Bokeh plot

    
    for i, ds in enumerate(datasets):
        try:
            # Load measurement data
            data = ds['measurement.hdf5'].xarray
            settings = ds['settings.json']
            
            # Find time coordinate
            coords = list(data.coords.keys())
            time_coord = next((c for c in coords if any(x in c.lower() for x in ['time', 'delay', 't1'])), coords[0])
            
            # Get signal data (assume first data variable contains T1 signal)
            signal_var = list(data.data_vars.keys())[0]
            signal_data = data[signal_var]
            
            # Extract time and signal values
            if len(signal_data.dims) == 1:
                time_vals = signal_data.coords[time_coord].values
                signal_vals = signal_data.values
                
                # Determine time units and convert appropriately
                # Check if time values are likely in seconds (very small values) or microseconds
                max_time = np.max(time_vals)
                if max_time < 1e-3:  # If max time < 1ms, assume it's in seconds
                    time_us = time_vals * 1e6  # Convert seconds to microseconds
                    t1_conversion = 1e6  # T1 will be in seconds, convert to microseconds
                else:  # Assume already in microseconds
                    time_us = time_vals
                    t1_conversion = 1  # T1 already in microseconds
                
                # Fit T1
                t1_value = fit_t1(time_vals, signal_vals)
                
                # Store trace data for Bokeh plot
                trace_data = {
                    'time': time_us,
                    'signal': signal_vals,
                    'sample_name': f'Sample_{i+1}',
                    'dataset_id': str(ds.uuid).replace('-', ''),
                    'T1_us': t1_value * t1_conversion if not np.isnan(t1_value) else np.nan
                }
                all_traces.append(trace_data)
                
                # Create result entry
                result = {
                    'dataset_id': ds.uuid,
                    'sample_name': f'Sample_{i+1}',
                    'T1_us': t1_value * t1_conversion if not np.isnan(t1_value) else np.nan
                }
                
                # Extract parameters from settings
                if settings:
                    for key, value in settings.items():
                        if isinstance(value, (int, float)) and 'param' in key.lower():
                            result[key] = value
                        elif key.startswith('param_'):
                            result[key] = value
                
                all_results.append(result)
            
        except Exception as e:
            print(f"Error processing dataset {i+1}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No valid T1 data found!")
        return None, None
    
    # Remove rows with NaN T1 values
    df_clean = df.dropna(subset=['T1_us'])
    
    # Get parameter columns
    param_cols = [col for col in df_clean.columns if 
                  col.startswith(('param_', 'attr_')) and 
                  df_clean[col].dtype in ['float64', 'int64', 'object']]
    
    
    # Create correlation plots
    correlation_plots = []
    
    # Correlation matrix for numeric parameters
    numeric_params = [col for col in param_cols if df_clean[col].dtype in ['float64', 'int64']]
    
    if numeric_params:
        # Calculate correlations
        corr_data = df_clean[['T1_us'] + numeric_params].corr()['T1_us'].drop('T1_us')
        
        # Correlation bar plot
        if not corr_data.empty:
            fig, ax = plt.subplots(figsize=(8, 4))  # Made smaller
            corr_data.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel('Correlation Coefficient')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            correlation_plots.append(('Correlation Matrix', plot_to_b64(fig)))
        
        # Create combined scatter plots in a single figure
        valid_params = [param for param in numeric_params[:4] if df_clean[param].nunique() > 1]  # Limit to 4 plots
        
        if valid_params:
            # Always arrange plots in a single row
            n_plots = len(valid_params)
            rows, cols = 1, n_plots
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4))
            if n_plots == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, param in enumerate(valid_params):
                ax = axes[i]
                ax.scatter(df_clean[param], df_clean['T1_us'], alpha=0.7, s=40)
                ax.set_xlabel(param)
                ax.set_ylabel('T1 (μs)')
                ax.set_title(f'T1 vs {param}')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(df_clean) > 2:
                    z = np.polyfit(df_clean[param], df_clean['T1_us'], 1)
                    p = np.poly1d(z)
                    ax.plot(df_clean[param], p(df_clean[param]), "r--", alpha=0.8)
                    
                    # Add correlation coefficient
                    corr_coef = df_clean[param].corr(df_clean['T1_us'])
                    ax.text(0.05, 0.95, f'R = {corr_coef:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            correlation_plots.append(('T1 vs Parameters', plot_to_b64(fig)))
    
    # Create Bokeh plot with all T1 traces
    def create_bokeh_t1_traces():
        if not all_traces:
            return ""
        
        # Create Bokeh figure
        p = figure(
            width=800, 
            height=600,
            x_axis_label="Time (μs)",
            y_axis_label="Signal",
            toolbar_location="above"
        )
        p.toolbar.logo = None
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Sample", "@sample_name"),
            ("Dataset ID", "@dataset_id"),
            ("T1", "@T1_us{0.2f} μs"),
            ("Time", "@x{0.2f} μs"),
            ("Signal", "@y{0.4f}")
        ])
        p.add_tools(hover)
        
        # Plot each trace with different colors
        colors = Category10[10]
        for i, trace in enumerate(all_traces):
            if not np.isnan(trace['T1_us']):  # Only plot traces with valid T1 fits
                color = colors[i % len(colors)]
                
                # Create data source with metadata for hover
                source_data = {
                    'x': trace['time'],
                    'y': trace['signal'],
                    'sample_name': [trace['sample_name']] * len(trace['time']),
                    'dataset_id': [trace['dataset_id']] * len(trace['time']),
                    'T1_us': [trace['T1_us']] * len(trace['time'])
                }
                
                p.line(
                    'x', 'y', 
                    source=source_data,
                    legend_label=f"{trace['sample_name']} (T1={trace['T1_us']:.1f}μs)",
                    line_color=color,
                    line_width=2,
                    alpha=0.8
                )
                
                p.scatter(
                    'x', 'y',
                    source=source_data,
                    color=color,
                    size=4,
                    alpha=0.6
                )
        
        # Configure legend
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "8pt"
        
        # Generate script and div components
        script, div = components(p)
        return script, div
    
    bokeh_script, bokeh_div = create_bokeh_t1_traces() if all_traces else ("", "")
    
    # Convert UUIDs to format without dashes
    def format_uuid_no_dash(uuid_str):
        return str(uuid_str).replace('-', '')
    
    # Create a copy of the dataframe for display with formatted UUIDs
    df_display = df_clean.copy()
    df_display['dataset_id'] = df_display['dataset_id'].apply(format_uuid_no_dash)
    
    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>T1 Correlation Analysis Report</title>
        {CDN.render_css()}
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .summary {{ background-color: #e7f3ff; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; }}
            .plot-row {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0; padding: 0 20px; }}
            .plot-item {{ flex: 1; min-width: 300px; margin: 10px; text-align: center; }}
            .plot-item img {{ max-width: 100%; height: auto; }}
            .correlation-matrix {{ text-align: center; margin: 20px 0; }}
            .correlation-matrix img {{ max-width: 50%; width: 50%; height: auto; }}
            .correlation-scatter {{ text-align: center; margin: 20px 0; }}
            .correlation-scatter img {{ max-width: 100%; width: 100%; height: auto; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>T1 Correlation Analysis Report</h1>
        
        <div class="summary">
            <h3>Analysis Summary</h3>
            <p><strong>Total datasets analyzed:</strong> {len(datasets)}</p>
            <p><strong>Valid T1 measurements:</strong> {len(df_clean)}</p>
            <p><strong>Mean T1:</strong> {df_clean['T1_us'].mean():.2f} ± {df_clean['T1_us'].std():.2f} μs</p>
            <p><strong>T1 Range:</strong> {df_clean['T1_us'].min():.2f} - {df_clean['T1_us'].max():.2f} μs</p>
            <p><strong>Parameters found:</strong> {', '.join(param_cols)}</p>
        </div>
        
        <h2>All T1 Traces</h2>
        <div style="margin: 20px 0;">
        {bokeh_div if bokeh_div else '<p>No T1 traces available for plotting.</p>'}
        </div>
        
        <h2>Correlation Analysis</h2>
    """
    
    # Add correlation plots with proper layout
    if correlation_plots:
        # Separate correlation matrix from scatter plots
        correlation_matrix_plot = None
        scatter_plots = []
        
        for title, plot in correlation_plots:
            if 'Correlation Matrix' in title:
                correlation_matrix_plot = (title, plot)
            else:
                scatter_plots.append((title, plot))
        
        # Add correlation matrix plot first (centered, above scatter plots)
        if correlation_matrix_plot:
            title, plot = correlation_matrix_plot
            html += f"""
            <div class="correlation-matrix">
                <h3>{title}</h3>
                <img src="{plot}" alt="{title}">
            </div>
            """
        
        # Add scatter plots (now combined in single image)
        if scatter_plots:
            for title, plot in scatter_plots:
                html += f"""
                <div class="correlation-scatter">
                    <h3>{title}</h3>
                    <img src="{plot}" alt="{title}">
                </div>
                """
    
    # Add the table at the end
    html += f"""
        <h2>T1 and Parameters Table</h2>
        {df_display.to_html(index=False, float_format=lambda x: f'{x:.3f}' if pd.notnull(x) else 'N/A')}
    """
    
    html += f"""
        {CDN.render_js()}
        {bokeh_script if bokeh_script else ''}
    </body>
    </html>
    """
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Report saved to: {output_file}")
    print(f"Analyzed {len(df_clean)} samples with T1 range: {df_clean['T1_us'].min():.1f} - {df_clean['T1_us'].max():.1f} μs")
    
    # Create a new dataset to store the analysis report
    analysis_dataset = dataset.create('T1 Correlation Analysis',
                                    scope_name=None,
                                    description=f'T1 correlation analysis report for {len(df_clean)} samples',
                                    tags=['T1_analysis', 'correlation_report'],
                                    # attributes={'analysis_type': 'T1_correlation', 'sample_count': len(df_clean)}
                                    )
    
    # Add the HTML report file to the dataset
    analysis_dataset['analysis.html'] = Path(output_file)
    return output_file, df_clean


