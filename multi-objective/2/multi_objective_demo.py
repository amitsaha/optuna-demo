"""
Multi-objective optimization demo using Optuna with XGBoost on UCI Adult Income dataset.
This demo optimizes for both accuracy and model size (number of trees).
"""
import numpy as np
import optuna
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', )))
from utils.data_utils import load_adult_income_data

scorers = {
    "accuracy": "accuracy",
    "neg_log_loss": make_scorer(log_loss, greater_is_better=False),
}

def objective(trial, X_train, y_train):
    """
    Multi-objective function for Optuna to optimize XGBoost hyperparameters.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        
    Returns:
        tuple: (accuracy, model_complexity) where complexity is negative n_estimators
               (we want to minimize complexity, so we negate it)
    """
    # Define hyperparameter search space
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': 1,  # To avoid nested parallelism
    }
    
    # Create XGBoost classifier
    model = XGBClassifier(**param)
    
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=3,
        scoring=scorers,
        return_train_score=True,
        n_jobs=-1,
    )

    # Objective 1: maximize accuracy
    mean_acc = np.mean(cv_results["test_accuracy"])

    train_loss = -cv_results["train_neg_log_loss"]
    val_loss = -cv_results["test_neg_log_loss"]
    # Objective 2: minimize generalization gap (val_loss - train_loss)
    mean_gap = np.mean(val_loss - train_loss)
    # # Store train_loss and val_loss for later plotting
    # trial.set_user_attr("train_loss", train_loss.tolist())
    # trial.set_user_attr("val_loss", val_loss.tolist())
    trial.set_user_attr("generalization_gap", mean_gap )
    
    return mean_acc, mean_gap

# Two key concepts for optuna:
# 1. Study - an optimization session
# 2. Trial - a single execution of the objective function
def run_multi_objective_optimization(n_trials=50):
    """
    Run multi-objective optimization study.
    
    Args:
        n_trials: Number of optimization trials
        
    Returns:
        study: Completed Optuna study object
    """
    # Load data
    print("Loading UCI Adult Income dataset...")
    X_train, X_test, y_train, y_test = load_adult_income_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create study with two objectives: maximize accuracy, minimize complexity
    study = optuna.create_study(
        directions=['maximize', 'minimize'],  # maximize accuracy, minimize generalization gap
        study_name='xgboost_multi_objective'
    )
    
    # Optimize
    print(f"\nStarting multi-objective optimization with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    # Print results
    print("\n" + "="*50)
    print("Multi-objective optimization completed!")
    print("="*50)
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of Pareto optimal solutions: {len(study.best_trials)}")
    
    print("\nPareto optimal solutions:")
    print("-" * 50)
    for i, trial in enumerate(study.best_trials):
        print(f"\nSolution {i+1}:")
        print(f"  Trial: {trial.number}")
        print(f"  Accuracy: {trial.values[0]:.4f}")
        print(f"  N_estimators: {trial.values[1]:.0f}")
        print(f"  Key parameters:")
        print(f"    - max_depth: {trial.params['max_depth']}")
        print(f"    - learning_rate: {trial.params['learning_rate']:.4f}")
        print(f"    - n_estimators: {trial.params['n_estimators']}")
    
    # Select a balanced solution (closest to middle of Pareto front)
    print("\n" + "="*50)
    print("Selecting a balanced solution from Pareto front...")
    
    if len(study.best_trials) > 0:
        # Find trial with median accuracy among Pareto optimal solutions
        sorted_trials = sorted(study.best_trials, key=lambda t: t.values[0])
        median_idx = len(sorted_trials) // 2
        selected_trial = sorted_trials[median_idx]
        
        print(f"Selected trial: {selected_trial.number}")
        print(f"Accuracy: {selected_trial.values[0]:.4f}")
        print(f"N_estimators: {selected_trial.values[1]:.0f}" )
        print(f"Generalization gap: {selected_trial.user_attrs['generalization_gap']:.4f}" )
        
        # Train final model with selected parameters
        print("\nTraining final model with selected parameters...")
        best_params = selected_trial.params.copy()
        best_params.update({
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
        })
        
        final_model = XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_accuracy = final_model.score(X_test, y_test)
        print(f"\nTest set accuracy: {test_accuracy:.4f}")
    
    return study


if __name__ == '__main__':
    study = run_multi_objective_optimization(n_trials=50)
