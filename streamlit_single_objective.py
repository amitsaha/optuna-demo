"""
Streamlit app for single-objective optimization using Optuna with XGBoost.
"""
import streamlit as st
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import pandas as pd
import plotly.graph_objects as go
from data_utils import load_adult_income_data, get_dataset_info


# Set page config
st.set_page_config(
    page_title="Single-Objective Optuna Demo",
    page_icon="🎯",
    layout="wide"
)


def objective(trial, X_train, y_train, param_config):
    """
    Objective function for Optuna to optimize XGBoost hyperparameters.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        param_config: Dictionary with parameter ranges
        
    Returns:
        float: Mean cross-validation accuracy score
    """
    param = {
        'max_depth': trial.suggest_int('max_depth', 
                                        param_config['max_depth'][0], 
                                        param_config['max_depth'][1]),
        'learning_rate': trial.suggest_float('learning_rate', 
                                              param_config['learning_rate'][0], 
                                              param_config['learning_rate'][1], 
                                              log=True),
        'n_estimators': trial.suggest_int('n_estimators', 
                                           param_config['n_estimators'][0], 
                                           param_config['n_estimators'][1]),
        'min_child_weight': trial.suggest_int('min_child_weight', 
                                               param_config['min_child_weight'][0], 
                                               param_config['min_child_weight'][1]),
        'gamma': trial.suggest_float('gamma', 
                                      param_config['gamma'][0], 
                                      param_config['gamma'][1]),
        'subsample': trial.suggest_float('subsample', 
                                          param_config['subsample'][0], 
                                          param_config['subsample'][1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 
                                                 param_config['colsample_bytree'][0], 
                                                 param_config['colsample_bytree'][1]),
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
    }
    
    model = XGBClassifier(**param)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()


# Title and description
st.title("🎯 Single-Objective Optimization with Optuna")
st.markdown("""
This demo shows how to use **Optuna** for single-objective hyperparameter optimization 
with **XGBoost** on the **UCI Adult Income** dataset.

The goal is to maximize the classification accuracy by finding the best hyperparameters.
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Configuration")

# Dataset info
with st.sidebar.expander("📊 Dataset Information", expanded=False):
    try:
        dataset_info = get_dataset_info()
        st.write(f"**Name:** {dataset_info['name']}")
        st.write(f"**Features:** {dataset_info['n_features']}")
        st.write(f"**Samples:** {dataset_info['n_samples']}")
    except Exception as e:
        st.write("Unable to fetch dataset info")

# Optimization parameters
st.sidebar.subheader("Optimization Settings")
n_trials = st.sidebar.slider("Number of trials", min_value=10, max_value=100, value=30, step=10)

# Hyperparameter ranges
st.sidebar.subheader("Hyperparameter Ranges")

col1, col2 = st.sidebar.columns(2)
with col1:
    max_depth_min = st.number_input("max_depth min", value=3, min_value=1, max_value=10)
with col2:
    max_depth_max = st.number_input("max_depth max", value=10, min_value=3, max_value=20)

col1, col2 = st.sidebar.columns(2)
with col1:
    lr_min = st.number_input("learning_rate min", value=0.01, min_value=0.001, max_value=0.5, format="%.3f")
with col2:
    lr_max = st.number_input("learning_rate max", value=0.3, min_value=0.01, max_value=1.0, format="%.2f")

col1, col2 = st.sidebar.columns(2)
with col1:
    n_est_min = st.number_input("n_estimators min", value=50, min_value=10, max_value=200)
with col2:
    n_est_max = st.number_input("n_estimators max", value=300, min_value=100, max_value=500)

col1, col2 = st.sidebar.columns(2)
with col1:
    mcw_min = st.number_input("min_child_weight min", value=1, min_value=1, max_value=5)
with col2:
    mcw_max = st.number_input("min_child_weight max", value=10, min_value=5, max_value=20)

col1, col2 = st.sidebar.columns(2)
with col1:
    gamma_min = st.number_input("gamma min", value=0.0, min_value=0.0, max_value=0.3, format="%.2f")
with col2:
    gamma_max = st.number_input("gamma max", value=0.5, min_value=0.1, max_value=1.0, format="%.2f")

col1, col2 = st.sidebar.columns(2)
with col1:
    subsample_min = st.number_input("subsample min", value=0.6, min_value=0.5, max_value=0.9, format="%.2f")
with col2:
    subsample_max = st.number_input("subsample max", value=1.0, min_value=0.7, max_value=1.0, format="%.2f")

col1, col2 = st.sidebar.columns(2)
with col1:
    colsample_min = st.number_input("colsample_bytree min", value=0.6, min_value=0.5, max_value=0.9, format="%.2f")
with col2:
    colsample_max = st.number_input("colsample_bytree max", value=1.0, min_value=0.7, max_value=1.0, format="%.2f")

# Run optimization button
if st.sidebar.button("🚀 Run Optimization", type="primary"):
    # Create parameter configuration
    param_config = {
        'max_depth': (max_depth_min, max_depth_max),
        'learning_rate': (lr_min, lr_max),
        'n_estimators': (n_est_min, n_est_max),
        'min_child_weight': (mcw_min, mcw_max),
        'gamma': (gamma_min, gamma_max),
        'subsample': (subsample_min, subsample_max),
        'colsample_bytree': (colsample_min, colsample_max),
    }
    
    # Load data
    with st.spinner("Loading dataset..."):
        X_train, X_test, y_train, y_test = load_adult_income_data()
    
    st.success(f"✓ Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Create and run study
    with st.spinner(f"Running optimization with {n_trials} trials... This may take a few minutes."):
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, param_config), 
            n_trials=n_trials,
            show_progress_bar=False
        )
    
    # Store results in session state
    st.session_state['study'] = study
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    
    st.success("✓ Optimization completed!")

# Display results
if 'study' in st.session_state:
    study = st.session_state['study']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 History", "🎯 Best Parameters", "🧪 Test Evaluation"])
    
    with tab1:
        st.header("Optimization Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Accuracy", f"{study.best_value:.4f}")
        with col2:
            st.metric("Best Trial", study.best_trial.number)
        with col3:
            st.metric("Total Trials", len(study.trials))
    
    with tab2:
        st.header("Optimization History")
        
        # Plot optimization history
        trials_data = []
        for trial in study.trials:
            if trial.value is not None:
                trials_data.append({
                    'Trial': trial.number,
                    'Accuracy': trial.value
                })
        
        df_trials = pd.DataFrame(trials_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trials['Trial'],
            y=df_trials['Accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add best value line
        fig.add_trace(go.Scatter(
            x=df_trials['Trial'],
            y=[study.best_value] * len(df_trials),
            mode='lines',
            name='Best Accuracy',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Accuracy",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Best Hyperparameters")
        
        # Display best parameters
        best_params_df = pd.DataFrame([
            {'Parameter': k, 'Value': v}
            for k, v in study.best_params.items()
        ])
        
        st.dataframe(best_params_df, use_container_width=True, hide_index=True)
        
        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(study)
            
            st.subheader("Parameter Importance")
            
            importance_df = pd.DataFrame([
                {'Parameter': k, 'Importance': v}
                for k, v in importance.items()
            ]).sort_values('Importance', ascending=False)
            
            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Parameter'],
                orientation='h',
                marker=dict(color='lightblue')
            ))
            
            fig_importance.update_layout(
                title="Hyperparameter Importance",
                xaxis_title="Importance",
                yaxis_title="Parameter",
                height=400
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.info("Parameter importance calculation requires at least 2 trials.")
    
    with tab4:
        st.header("Test Set Evaluation")
        
        if st.button("Evaluate on Test Set"):
            with st.spinner("Training final model with best parameters..."):
                # Train final model
                best_params = study.best_params.copy()
                best_params.update({
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                })
                
                final_model = XGBClassifier(**best_params)
                final_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                
                # Evaluate
                train_accuracy = final_model.score(st.session_state['X_train'], st.session_state['y_train'])
                test_accuracy = final_model.score(st.session_state['X_test'], st.session_state['y_test'])
            
            st.success("✓ Model evaluation completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_accuracy:.4f}")
            with col2:
                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance (Top 10)")
            
            feature_importance = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(final_model.feature_importances_))],
                'Importance': final_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig_feat = go.Figure()
            fig_feat.add_trace(go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker=dict(color='green')
            ))
            
            fig_feat.update_layout(
                title="Top 10 Most Important Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            
            st.plotly_chart(fig_feat, use_container_width=True)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Optimization' to start!")
