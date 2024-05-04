import plotly.express as px
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV


def train_gridsearch(estimator, params_grid, X_train, y_train):
    precision_scorer = make_scorer(precision_score, greater_is_better=True)
    recall_scorer = make_scorer(recall_score, greater_is_better=True)
    accuracy_scorer = make_scorer(accuracy_score, greater_is_better=True)
    f1_scorer = make_scorer(f1_score, greater_is_better=True)
    balance_accuracy_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)
    roc_auac_scorer = make_scorer(roc_auc_score, greater_is_better=True)

    modelGD = GridSearchCV(
        estimator,
        params_grid,
        cv=5,  # 200 for val
        n_jobs=4,
        scoring={
            'precision': precision_scorer,
            'recall': recall_scorer,
            'accuracy': accuracy_scorer,
            'f1': f1_scorer,
            'balance_accuracy': balance_accuracy_scorer,
            'roc_auac': roc_auac_scorer,
        },
        refit='recall',  # I will retrain the model considering recall, that configuration will try not to fail for FN (the loan is given to a bad client)
    )

    modelGD.fit(X_train, y_train)
    print('Best hyperparameters found: {}\n'.format(modelGD.best_params_))
    print('Best estimator found: {}\n'.format(modelGD.best_estimator_))
    print('Training Recall: {}\n'.format(round(modelGD.score(X_train, y_train), 4)))

    return modelGD


def get_validation_metrics(model):
    cv_results = model.cv_results_
    best_index = model.best_index_
    precision = cv_results['mean_test_precision'][best_index]
    recall = cv_results['mean_test_recall'][best_index]
    accuracy = cv_results['mean_test_accuracy'][best_index]
    f1 = cv_results['mean_test_f1'][best_index]
    balance_accuracy = cv_results['mean_test_balance_accuracy'][best_index]
    roc_auac = cv_results['mean_test_roc_auac'][best_index]
    # Printing validation results
    print('Validation Results')
    print('Precision: {} ± {}'.format(round(precision, 3), round(cv_results['std_test_precision'][best_index],4)))
    print('Recall: {} ± {}'.format(round(recall, 3),  round(cv_results['std_test_recall'][best_index],4)))
    print('Accuracy: {} ± {}'.format(round(accuracy, 3), round(cv_results['std_test_accuracy'][best_index],4)))
    print('F1: {} ± {}'.format(round(f1, 3), round(cv_results['std_test_f1'][best_index],4)))
    print(
        'Balance Accuracy: {} ± {}'.format(
            round(balance_accuracy, 3), round(cv_results['std_test_balance_accuracy'][best_index],4)
        )
    )
    print('ROC AUAC: {} ± {}'.format(round(roc_auac, 3), round(cv_results['std_test_roc_auac'][best_index],4)))


def plot_confusion_matrix(y_train, y_pred):
    cm = confusion_matrix(y_train, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="True", y="Pred", color="#"),
        x=['y\u0302=1', 'y\u0302=0'],
        y=['y=1', 'y=0'],
        color_continuous_scale='BuPu',
    )
    fig.update_xaxes(side="top")
    fig.update_traces(text=cm, texttemplate="%{text}")
    fig.update_layout(
        title='Confusion Matrix',
        title_x=0.5,
        title_y=0.99,
        width=500,  # Set the width of the figure (in pixels)
        height=500,  # Set the height of the figure (in pixels)
        autosize=True,  # Disable auto-sizing of the figure
    )
    fig.show()


def plot_pr_curve(estimator, X_train, y_train, name):
    display = PrecisionRecallDisplay.from_estimator(estimator, X_train, y_train, name=name, plot_chance_level=True)
    _ = display.ax_.set_title("Precision-Recall curve")
