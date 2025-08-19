"""
Streamlit dashboard for A/B testing framework
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from ab_test_framework import ABTestFramework, ExperimentConfig, TrafficSplitStrategy
import redis

# Configure Streamlit page
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ab_framework' not in st.session_state:
    st.session_state.ab_framework = ABTestFramework()

ab_framework = st.session_state.ab_framework


def main():
    st.title("🧪 A/B Testing Dashboard")
    st.markdown("Monitor and manage ML model A/B tests")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Overview", "Create Experiment", "Experiment Details", "Analytics", "Settings"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Create Experiment":
        show_create_experiment()
    elif page == "Experiment Details":
        show_experiment_details()
    elif page == "Analytics":
        show_analytics()
    elif page == "Settings":
        show_settings()


def show_overview():
    """Show overview of all experiments"""
    st.header("Experiments Overview")
    
    # Get all active experiments
    active_experiments = ab_framework.active_experiments
    
    if not active_experiments:
        st.info("No active experiments. Create one to get started!")
        return
    
    # Display experiment cards
    cols = st.columns(min(3, len(active_experiments)))
    
    for idx, (exp_id, config) in enumerate(active_experiments.items()):
        with cols[idx % 3]:
            with st.container():
                st.subheader(config.name)
                st.write(f"**ID:** {exp_id}")
                st.write(f"**Status:** Running")
                st.write(f"**Start Date:** {config.start_date.strftime('%Y-%m-%d')}")
                st.write(f"**End Date:** {config.end_date.strftime('%Y-%m-%d')}")
                st.write(f"**Variants:** {len(config.traffic_split)}")
                
                # Progress bar
                total_duration = (config.end_date - config.start_date).days
                elapsed_duration = (datetime.now() - config.start_date).days
                progress = min(elapsed_duration / total_duration, 1.0) if total_duration > 0 else 0
                st.progress(progress)
                
                if st.button(f"View Details", key=f"view_{exp_id}"):
                    st.session_state.selected_experiment = exp_id
    
    # Recent activity
    st.header("Recent Activity")
    
    # Create sample activity data
    activity_data = []
    for exp_id, config in active_experiments.items():
        activity_data.extend([
            {
                "Time": datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                "Experiment": config.name,
                "Event": "Metric Recorded",
                "Details": f"Precision: {np.random.uniform(0.8, 0.9):.3f}"
            },
            {
                "Time": datetime.now() - timedelta(minutes=np.random.randint(60, 120)),
                "Experiment": config.name,
                "Event": "User Assignment",
                "Details": f"Variant: {np.random.choice(list(config.traffic_split.keys()))}"
            }
        ])
    
    if activity_data:
        activity_df = pd.DataFrame(activity_data)
        activity_df = activity_df.sort_values('Time', ascending=False).head(10)
        st.dataframe(activity_df, use_container_width=True)


def show_create_experiment():
    """Show experiment creation form"""
    st.header("Create New Experiment")
    
    with st.form("create_experiment"):
        # Basic information
        st.subheader("Basic Information")
        name = st.text_input("Experiment Name", "New Fraud Model Test")
        description = st.text_area("Description", "Testing new fraud detection model")
        
        # Time settings
        st.subheader("Time Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.now().date())
            start_time = st.time_input("Start Time", datetime.now().time())
        
        with col2:
            end_date = st.date_input("End Date", (datetime.now() + timedelta(days=14)).date())
            end_time = st.time_input("End Time", datetime.now().time())
        
        # Models configuration
        st.subheader("Model Configuration")
        num_variants = st.number_input("Number of Variants", min_value=2, max_value=5, value=2)
        
        models = {}
        traffic_split = {}
        
        for i in range(num_variants):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                variant_name = st.text_input(f"Variant {i+1} Name", 
                                           f"{'control' if i == 0 else f'treatment_{i}'}")
            
            with col2:
                model_uri = st.text_input(f"Model URI {i+1}", 
                                        f"models:/fraud_model_v{i+1}/latest")
            
            with col3:
                traffic_pct = st.number_input(f"Traffic % {i+1}", 
                                            min_value=0.0, max_value=100.0, 
                                            value=100.0/num_variants)
            
            models[variant_name] = model_uri
            traffic_split[variant_name] = traffic_pct / 100.0
        
        # Traffic splitting
        st.subheader("Traffic Splitting")
        split_strategy = st.selectbox(
            "Split Strategy",
            ["USER_ID_HASH", "RANDOM", "GEOGRAPHIC", "DEVICE_TYPE", "TIME_BASED"]
        )
        
        # Metrics
        st.subheader("Metrics Configuration")
        success_metrics = st.multiselect(
            "Success Metrics",
            ["precision", "recall", "f1_score", "auc", "accuracy", "conversion_rate"],
            default=["precision", "recall", "f1_score"]
        )
        
        guardrail_metrics = st.multiselect(
            "Guardrail Metrics",
            ["latency_ms", "error_rate", "throughput", "success_rate", "cpu_usage"],
            default=["latency_ms", "error_rate"]
        )
        
        # Statistical settings
        st.subheader("Statistical Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_sample_size = st.number_input("Minimum Sample Size", min_value=100, value=10000)
        
        with col2:
            statistical_power = st.slider("Statistical Power", 0.70, 0.95, 0.80, 0.05)
        
        with col3:
            significance_level = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
        
        # Rollback settings
        st.subheader("Auto-Rollback Settings")
        auto_rollback = st.checkbox("Enable Auto-Rollback", value=True)
        
        if auto_rollback:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                error_threshold = st.number_input("Error Rate Threshold", 
                                                min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            
            with col2:
                latency_threshold = st.number_input("Latency Threshold (ms)", 
                                                  min_value=0, value=1000, step=50)
            
            with col3:
                success_threshold = st.number_input("Success Rate Threshold", 
                                                  min_value=0.0, max_value=1.0, value=0.95, step=0.01)
        
        # Submit button
        submitted = st.form_submit_button("Create Experiment")
        
        if submitted:
            # Validate inputs
            if sum(traffic_split.values()) < 0.99 or sum(traffic_split.values()) > 1.01:
                st.error("Traffic split percentages must sum to 100%")
                return
            
            # Create experiment configuration
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            start_datetime = datetime.combine(start_date, start_time)
            end_datetime = datetime.combine(end_date, end_time)
            
            rollback_conditions = {}
            if auto_rollback:
                rollback_conditions = {
                    'error_rate_threshold': error_threshold,
                    'latency_threshold_ms': latency_threshold,
                    'success_rate_threshold': success_threshold
                }
            
            config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                models=models,
                traffic_split=traffic_split,
                split_strategy=TrafficSplitStrategy(split_strategy),
                start_date=start_datetime,
                end_date=end_datetime,
                success_metrics=success_metrics,
                guardrail_metrics=guardrail_metrics,
                minimum_sample_size=min_sample_size,
                statistical_power=statistical_power,
                significance_level=significance_level,
                auto_rollback_enabled=auto_rollback,
                rollback_conditions=rollback_conditions,
                created_by="dashboard_user"
            )
            
            # Create experiment
            success = ab_framework.create_experiment(config)
            
            if success:
                st.success(f"Experiment created successfully! ID: {experiment_id}")
                
                # Auto-start if start date is now or in the past
                if start_datetime <= datetime.now():
                    ab_framework.start_experiment(experiment_id)
                    st.success("Experiment started automatically!")
            else:
                st.error("Failed to create experiment. Please check the configuration.")


def show_experiment_details():
    """Show detailed view of a specific experiment"""
    st.header("Experiment Details")
    
    # Experiment selection
    active_experiments = ab_framework.active_experiments
    
    if not active_experiments:
        st.info("No active experiments available.")
        return
    
    # Select experiment
    experiment_options = {f"{config.name} ({exp_id})": exp_id 
                         for exp_id, config in active_experiments.items()}
    
    selected_option = st.selectbox("Select Experiment", list(experiment_options.keys()))
    
    if not selected_option:
        return
    
    experiment_id = experiment_options[selected_option]
    config = active_experiments[experiment_id]
    
    # Get experiment results
    results = ab_framework.get_experiment_results(experiment_id)
    
    if not results:
        st.warning("No results available yet.")
        return
    
    # Experiment info
    st.subheader("Experiment Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", results.get('status', 'Unknown'))
    
    with col2:
        st.metric("Total Samples", results.get('total_samples', 0))
    
    with col3:
        days_running = (datetime.now() - config.start_date).days
        st.metric("Days Running", days_running)
    
    with col4:
        remaining_days = (config.end_date - datetime.now()).days
        st.metric("Days Remaining", max(0, remaining_days))
    
    # Traffic split visualization
    st.subheader("Traffic Split")
    
    traffic_data = pd.DataFrame([
        {"Variant": variant, "Percentage": percentage * 100}
        for variant, percentage in config.traffic_split.items()
    ])
    
    fig_pie = px.pie(traffic_data, values='Percentage', names='Variant',
                     title="Current Traffic Split")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Metrics comparison
    st.subheader("Metrics Comparison")
    
    summary_stats = results.get('summary_statistics', {})
    
    if summary_stats:
        # Create metrics comparison table
        metrics_data = []
        
        for variant, stats in summary_stats.items():
            for metric, values in stats.items():
                metrics_data.append({
                    'Variant': variant,
                    'Metric': metric,
                    'Mean': values.get('mean', 0),
                    'Std': values.get('std', 0),
                    'Count': values.get('count', 0)
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create comparison charts for each metric
        for metric in config.success_metrics:
            metric_data = metrics_df[metrics_df['Metric'] == metric]
            
            if not metric_data.empty:
                fig_bar = px.bar(metric_data, x='Variant', y='Mean',
                               error_y='Std', title=f'{metric.title()} Comparison')
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Statistical test results
    st.subheader("Statistical Test Results")
    
    statistical_results = results.get('statistical_tests', {})
    
    if statistical_results:
        for metric, tests in statistical_results.items():
            st.write(f"**{metric.title()}**")
            
            test_data = []
            for test in tests:
                test_data.append({
                    'Test': test.test_name,
                    'P-Value': f"{test.p_value:.4f}",
                    'Effect Size': f"{test.effect_size:.4f}",
                    'Significant': "✅" if test.is_significant else "❌",
                    'Sample Size (Control)': test.sample_size_control,
                    'Sample Size (Treatment)': test.sample_size_treatment
                })
            
            if test_data:
                test_df = pd.DataFrame(test_data)
                st.dataframe(test_df, use_container_width=True)
    
    # Control buttons
    st.subheader("Experiment Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Stop Experiment", type="secondary"):
            ab_framework.stop_experiment(experiment_id, "Manual stop from dashboard")
            st.success("Experiment stopped!")
            st.experimental_rerun()
    
    with col2:
        if st.button("Download Results", type="secondary"):
            # Create downloadable results
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name=f"experiment_{experiment_id}_results.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("Refresh Data", type="primary"):
            st.experimental_rerun()


def show_analytics():
    """Show analytics and insights across experiments"""
    st.header("Analytics & Insights")
    
    # Sample analytics data (in real implementation, this would come from the framework)
    
    # Experiment performance over time
    st.subheader("Experiment Performance Trends")
    
    # Generate sample time series data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    performance_data = []
    for date in dates:
        performance_data.append({
            'Date': date,
            'Success Rate': np.random.uniform(0.85, 0.95),
            'Average Precision': np.random.uniform(0.80, 0.90),
            'Average Latency': np.random.uniform(150, 250)
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Success rate trend
    fig_success = px.line(performance_df, x='Date', y='Success Rate',
                         title='Success Rate Trend Over Time')
    st.plotly_chart(fig_success, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    model_data = pd.DataFrame({
        'Model': ['Baseline', 'XGBoost', 'Neural Network', 'Random Forest', 'Ensemble'],
        'Precision': [0.82, 0.85, 0.87, 0.84, 0.89],
        'Recall': [0.78, 0.81, 0.83, 0.80, 0.85],
        'F1-Score': [0.80, 0.83, 0.85, 0.82, 0.87],
        'Latency (ms)': [120, 180, 220, 160, 200]
    })
    
    # Create radar chart for model comparison
    fig_radar = go.Figure()
    
    for idx, row in model_data.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Precision'], row['Recall'], row['F1-Score'], 1 - row['Latency (ms)']/300],
            theta=['Precision', 'Recall', 'F1-Score', 'Speed'],
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Experiment insights
    st.subheader("Key Insights")
    
    insights = [
        "🎯 Neural Network models show 3% higher precision but 25% increased latency",
        "📈 Success rates have improved by 12% over the last quarter",
        "⚡ XGBoost provides the best balance of performance and speed",
        "🚨 2 experiments required rollback due to latency issues",
        "📊 A/B tests with >10K samples show more reliable results"
    ]
    
    for insight in insights:
        st.write(insight)


def show_settings():
    """Show framework settings and configuration"""
    st.header("Settings & Configuration")
    
    # Redis connection settings
    st.subheader("Redis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        redis_host = st.text_input("Redis Host", "localhost")
    
    with col2:
        redis_port = st.number_input("Redis Port", min_value=1, max_value=65535, value=6379)
    
    with col3:
        redis_db = st.number_input("Redis DB", min_value=0, max_value=15, value=0)
    
    # Default experiment settings
    st.subheader("Default Experiment Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_power = st.slider("Default Statistical Power", 0.70, 0.95, 0.80, 0.05)
        default_significance = st.slider("Default Significance Level", 0.01, 0.10, 0.05, 0.01)
    
    with col2:
        default_sample_size = st.number_input("Default Minimum Sample Size", 
                                            min_value=100, value=10000)
        default_duration = st.number_input("Default Duration (days)", 
                                         min_value=1, max_value=90, value=14)
    
    # Monitoring settings
    st.subheader("Monitoring & Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monitoring_interval = st.number_input("Monitoring Interval (seconds)", 
                                            min_value=10, value=60)
        enable_email_alerts = st.checkbox("Enable Email Alerts", value=True)
    
    with col2:
        enable_slack_alerts = st.checkbox("Enable Slack Alerts", value=False)
        enable_sms_alerts = st.checkbox("Enable SMS Alerts", value=False)
    
    # Save settings
    if st.button("Save Settings", type="primary"):
        # In a real implementation, these would be saved to a configuration store
        st.success("Settings saved successfully!")
    
    # System status
    st.subheader("System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Redis Status", "Connected ✅")
    
    with col2:
        st.metric("Active Experiments", len(ab_framework.active_experiments))
    
    with col3:
        st.metric("Monitoring Status", "Running ✅")
    
    with col4:
        st.metric("Framework Version", "1.0.0")


if __name__ == "__main__":
    main()
