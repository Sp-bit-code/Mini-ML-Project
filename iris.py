# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# st.set_page_config(page_title="🌸 Iris Classifier", layout="centered")

# # --- Helper Functions ---
# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path)
#     # Ensure 'Id' column is dropped if it exists and remove duplicates
#     df.drop(columns=["Id"], errors='ignore', inplace=True)
#     df.drop_duplicates(inplace=True)
#     return df

# # Function to generate a more realistic and separable dummy Iris dataset
# def generate_realistic_dummy_iris():
#     data = []
#     # Using more distinct means and slightly tighter standard deviations
#     # based on approximate real Iris dataset characteristics for better separability
    
#     # Iris-setosa (50 samples)
#     for _ in range(50):
#         data.append([
#             np.random.normal(5.0, 0.2), # SepalLengthCm (e.g., 4.8-5.2)
#             np.random.normal(3.4, 0.2), # SepalWidthCm (e.g., 3.2-3.6)
#             np.random.normal(1.4, 0.1), # PetalLengthCm (e.g., 1.3-1.5) - VERY DISTINCT
#             np.random.normal(0.2, 0.05),# PetalWidthCm (e.g., 0.15-0.25) - VERY DISTINCT
#             'Iris-setosa'
#         ])
#     # Iris-versicolor (50 samples)
#     for _ in range(50):
#         data.append([
#             np.random.normal(5.9, 0.3), # SepalLengthCm (e.g., 5.6-6.2)
#             np.random.normal(2.7, 0.2), # SepalWidthCm (e.g., 2.5-2.9)
#             np.random.normal(4.2, 0.3), # PetalLengthCm (e.g., 3.9-4.5)
#             np.random.normal(1.3, 0.1), # PetalWidthCm (e.g., 1.2-1.4)
#             'Iris-versicolor'
#         ])
#     # Iris-virginica (50 samples)
#     for _ in range(50):
#         data.append([
#             np.random.normal(6.5, 0.4), # SepalLengthCm (e.g., 6.1-6.9)
#             np.random.normal(3.0, 0.2), # SepalWidthCm (e.g., 2.8-3.2)
#             np.random.normal(5.5, 0.4), # PetalLengthCm (e.g., 5.1-5.9)
#             np.random.normal(2.0, 0.1), # PetalWidthCm (e.g., 1.9-2.1)
#             'Iris-virginica'
#         ])

#     df = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
#     # Add some duplicates for robustness check
#     df = pd.concat([df, df.sample(5, random_state=42)], ignore_index=True)
#     df.to_csv("Iris.csv", index=False)
#     return df

# # --- Streamlit App ---
# st.title("🌸 Iris Flower Classification using Decision Tree")

# st.markdown("""
# This application classifies Iris flowers into their respective species (Setosa, Versicolor, Virginica)
# using a Decision Tree model. Follow the steps below:
# 1. Load the Iris dataset.
# 2. Train the Decision Tree model.
# 3. Evaluate the model's performance.
# 4. Predict the species of a new Iris flower sample.
# """)

# # --- Step 1: Load Data ---
# st.header("1. Load Dataset")
# st.markdown("""
# **⭐ IMPORTANT: For best results and high accuracy, please use the actual Iris dataset.**

# You can download it from many sources, for example:
# - **UCI Machine Learning Repository:** [https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris)
# - Directly from scikit-learn (programmatically): `from sklearn.datasets import load_iris; iris_data = load_iris(as_frame=True); df = iris_data.frame; df.rename(columns={'sepal length (cm)': 'SepalLengthCm', 'sepal width (cm)': 'SepalWidthCm', 'petal length (cm)': 'PetalLengthCm', 'petal width (cm)': 'PetalWidthCm', 'target': 'Species'}, inplace=True); df['Species'] = df['Species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})`
# """)

# file_path = st.text_input("Enter file path (e.g., Iris.csv)", value="Iris.csv") # Default to Iris.csv

# if st.button("Load Data"):
#     df = None # Initialize df to None

#     if file_path == "Iris.csv":
#         try:
#             st.info("No custom file path entered. A **realistic dummy 'Iris.csv'** has been generated for demonstration purposes. Please note that for true high accuracy, an actual Iris dataset is recommended.")
#             df = generate_realistic_dummy_iris()
#         except Exception as e:
#             st.warning(f"Could not create dummy Iris.csv: {e}")
#     else:
#         try:
#             df = load_data(file_path)
#         except FileNotFoundError:
#             st.error(f"❌ Error: File not found at '{file_path}'. Please check the path.")
#         except Exception as e:
#             st.error(f"❌ An error occurred during data loading: {e}")

#     if df is not None: # Proceed only if df was successfully loaded/generated
#         if df.empty:
#             st.warning("The dataset loaded is empty after cleaning. Please check your CSV file.")
#         elif "Species" not in df.columns:
#             st.error("The 'Species' column was not found in the dataset. Please ensure your dataset has a 'Species' column (case-sensitive).")
#         else:
#             X = df.drop("Species", axis=1)
#             y = df["Species"]

#             # Ensure consistent column names after dropping 'Id' for the scaler
#             X.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


#             # Stratified split to maintain class proportions in train and test sets
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, stratify=y, random_state=42)

#             # Data Scaling: Fit scaler ONLY on training data, then transform both train and test
#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(X_train)
#             X_test_scaled = scaler.transform(X_test)

#             # Convert scaled arrays back to DataFrames to retain column names for model training/plotting
#             X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#             X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

#             st.session_state.df = df
#             st.session_state.X_train = X_train
#             st.session_state.X_test = X_test
#             st.session_state.y_train = y_train
#             st.session_state.y_test = y_test
#             st.session_state.scaler = scaler # Store scaler for future predictions

#             st.success(f"✅ Dataset Loaded, Cleaned, Scaled, and Split 80/20. Total samples: {len(df)}")
#             st.subheader("First 5 rows of the processed dataset:")
#             st.dataframe(df.head())
#             st.subheader("Distribution of Species:")
#             st.bar_chart(df['Species'].value_counts())

# # --- Step 2: Train Model ---
# if "X_train" in st.session_state:
#     st.header("2. Train Model")
#     st.write("Configure Decision Tree Hyperparameters (Optional):")

#     # Default value for max_depth should be an integer for the slider
#     max_depth = st.slider("Max Depth of Tree", min_value=1, max_value=15, value=5, help="The maximum depth of the tree. A value around 3-7 is often good for Iris to balance bias and variance.")
#     min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, help="The minimum number of samples required to split an internal node.")
#     min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, help="The minimum number of samples required to be at a leaf node.")
#     criterion = st.selectbox("Criterion", ("gini", "entropy"), help="The function to measure the quality of a split.")

#     if st.button("Train Model"):
#         model = DecisionTreeClassifier(
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             criterion=criterion,
#             random_state=42
#         )
#         model.fit(st.session_state.X_train, st.session_state.y_train)
#         st.session_state.model = model
#         st.success("🎯 Model Trained with selected hyperparameters!")

#         st.subheader("Decision Tree Visualization:")
#         # Plotting the tree (requires matplotlib 3.3.0+)
#         fig, ax = plt.subplots(figsize=(20, 10))
#         plot_tree(model, feature_names=st.session_state.X_train.columns, class_names=model.classes_, filled=True, rounded=True, ax=ax)
#         st.pyplot(fig)
#         st.caption("A visualization of the trained Decision Tree. Each node shows the decision rule and class distribution.")

# # --- Step 3: Evaluation ---
# if "model" in st.session_state:
#     st.header("3. Model Evaluation")
#     st.write("This section provides a comprehensive evaluation of the trained model.")

#     model = st.session_state.model
#     X_train = st.session_state.X_train
#     X_test = st.session_state.X_test
#     y_train = st.session_state.y_train
#     y_test = st.session_state.y_test

#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)

#     # Calculate metrics
#     train_acc = accuracy_score(y_train, y_train_pred)
#     test_acc = accuracy_score(y_test, y_test_pred)
#     precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
#     recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
#     f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

#     col1, col2, col3 = st.columns(3)
#     col1.metric("✅ Train Accuracy", f"{train_acc*100:.2f}%")
#     col2.metric("📊 Test Accuracy", f"{test_acc*100:.2f}%")
#     col3.metric("🎯 Precision", f"{precision*100:.2f}%")

#     col4, col5, col6 = st.columns(3)
#     col4.metric("📈 Recall", f"{recall*100:.2f}%")
#     col5.metric("🏆 F1 Score", f"{f1*100:.2f}%")
#     col6.metric("🌳 Tree Depth", model.get_depth())


#     st.subheader("Confusion Matrix")
#     cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
#     fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=model.classes_, yticklabels=model.classes_, ax=ax_cm)
#     ax_cm.set_xlabel("Predicted")
#     ax_cm.set_ylabel("Actual")
#     ax_cm.set_title("Confusion Matrix (Test Set)")
#     st.pyplot(fig_cm)
#     st.caption("This matrix shows the number of correct and incorrect predictions made by the classification model compared to the actual outcomes. Diagonal values represent correct predictions.")

#     st.subheader("📉 Generalization Plot")
#     fig_gen, ax_gen = plt.subplots(figsize=(8, 5))
#     metrics_df = pd.DataFrame({
#         'Metric': ['Train Accuracy', 'Test Accuracy', 'Generalization Gap'],
#         'Value': [train_acc, test_acc, abs(train_acc - test_acc)]
#     })
#     sns.barplot(x='Metric', y='Value', data=metrics_df, palette=['green', 'orange', 'red'], ax=ax_gen)
#     ax_gen.set_ylim(0, 1.05)
#     ax_gen.set_ylabel("Accuracy/Difference")
#     ax_gen.set_title("Train vs Test Accuracy & Generalization Gap")
#     st.pyplot(fig_gen)
#     st.caption("The generalization plot visualizes how well the model performs on unseen data compared to the training data. A large 'Generalization Gap' suggests overfitting.")


#     st.subheader("Cross-Validation Scores")
#     st.write("Cross-validation helps in getting a more reliable estimate of model performance and detecting overfitting.")
#     try:
#         # Use KFold for cross-validation on the entire dataset
#         # Note: When doing cross-validation, it's typical to use the full dataset (X, y)
#         # However, for consistency with train/test splits, we concatenate the scaled data here.
#         # Ensure that the scaler is re-applied correctly if you use raw X, y for KFold.
#         X_full_scaled = pd.concat([X_train, X_test])
#         y_full = pd.concat([y_train, y_test])
        
#         cv = KFold(n_splits=5, shuffle=True, random_state=42)
#         cv_scores = cross_val_score(model, X_full_scaled, y_full, cv=cv, scoring='accuracy')

#         st.info(f"Mean Cross-Validation Accuracy: **{np.mean(cv_scores)*100:.2f}%** (Std Dev: {np.std(cv_scores)*100:.2f}%)")
#         st.write("Individual Cross-Validation Scores:")
#         st.write(cv_scores)
#     except Exception as e:
#         st.error(f"Error during cross-validation: {e}")

# # --- Step 4: Prediction ---
# if "model" in st.session_state and "scaler" in st.session_state:
#     st.header("4. Predict New Sample")
#     st.write("Enter the measurements for a new Iris flower to predict its species.")

#     col1, col2 = st.columns(2)
#     with col1:
#         sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, step=0.1, format="%.1f")
#         sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, step=0.1, format="%.1f")
#     with col2:
#         petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, step=0.1, format="%.1f")
#         petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1, format="%.1f")

#     if st.button("Predict Species"):
#         try:
#             # Create a DataFrame for the single sample
#             sample_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
#                                   columns=st.session_state.X_train.columns) # Use X_train.columns for consistency
#             # Scale the new sample using the *trained* scaler
#             sample_scaled = st.session_state.scaler.transform(sample_df)
#             prediction = st.session_state.model.predict(sample_scaled)
#             prediction_proba = st.session_state.model.predict_proba(sample_scaled)

#             st.success(f"🔍 Predicted Species: **{prediction[0]}**")
#             st.subheader("Prediction Probabilities:")
#             proba_df = pd.DataFrame(prediction_proba, columns=st.session_state.model.classes_)
#             st.dataframe(proba_df.T.rename(columns={0: 'Probability'}).sort_values(by='Probability', ascending=False))

#         except Exception as e:
#             st.error(f"Error during prediction: {e}")

# st.markdown("---")
# st.caption("Developed with ❤️ using Streamlit and Scikit-learn.")









































# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# from sklearn.pipeline import Pipeline 
# st.set_page_config(page_title="🌸 Iris Classifier", layout="centered")
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'X_train' not in st.session_state:
#     st.session_state.X_train = None
# if 'X_test' not in st.session_state:
#     st.session_state.X_test = None
# if 'y_train' not in st.session_state:
#     st.session_state.y_train = None
# if 'y_test' not in st.session_state:
#     st.session_state.y_test = None
# if 'scaler' not in st.session_state:
#     st.session_state.scaler = None
# if 'model' not in st.session_state:
#     st.session_state.model = None
# if 'pipeline_model' not in st.session_state:
#     st.session_state.pipeline_model = None
# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path)
#     df.drop(columns=["Id"], errors='ignore', inplace=True)
#     df.drop_duplicates(inplace=True)
#     return df
# def generate_realistic_dummy_iris():
#     data = []
#     for _ in range(50):
#         data.append([
#             np.random.normal(5.0, 0.2), 
#             np.random.normal(3.4, 0.2), 
#             np.random.normal(1.4, 0.1),
#             np.random.normal(0.2, 0.05),
#             'Iris-setosa'
#         ])
#     # Iris-versicolor (50 samples)
#     for _ in range(50):
#         data.append([
#             np.random.normal(5.9, 0.3), 
#             np.random.normal(2.7, 0.2), 
#             np.random.normal(4.2, 0.3), 
#             np.random.normal(1.3, 0.1), 
#             'Iris-versicolor'
#         ])
#     for _ in range(50):
#         data.append([
#             np.random.normal(6.5, 0.4), # SepalLengthCm (e.g., 6.1-6.9)
#             np.random.normal(3.0, 0.2), # SepalWidthCm (e.g., 2.8-3.2)
#             np.random.normal(5.5, 0.4), # PetalLengthCm (e.g., 5.1-5.9)
#             np.random.normal(2.0, 0.1), # PetalWidthCm (e.g., 1.9-2.1)
#             'Iris-virginica'
#         ])

#     df = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
#     df = pd.concat([df, df.sample(5, random_state=42)], ignore_index=True)
#     df.to_csv("Iris.csv", index=False)
#     return df
# st.title("🌸 Iris Flower Classification using Decision Tree")

# st.markdown("""
# This application classifies Iris flowers into their respective species (Setosa, Versicolor, Virginica)
# using a Decision Tree model. Follow the steps below:
# 1. Load the Iris dataset.
# 2. Train the Decision Tree model.
# 3. Evaluate the model's performance.
# 4. Predict the species of a new Iris flower sample.
# """)

# st.header("1. Load Dataset")
# st.markdown("""
# **⭐ IMPORTANT: For best results and high accuracy, please use the actual Iris dataset.**

# You can download it from many sources, for example:
# - Directly from scikit-learn (programmatically): `from sklearn.datasets import load_iris; iris_data = load_iris(as_frame=True); df = iris_data.frame; df.rename(columns={'sepal length (cm)': 'SepalLengthCm', 'sepal width (cm)': 'SepalWidthCm', 'petal length (cm)': 'PetalLengthCm', 'petal width (cm)': 'PetalWidthCm', 'target': 'Species'}, inplace=True); df['Species'] = df['Species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})`
# """)

# file_path = st.text_input("Enter file path (e.g., Iris.csv)", value="Iris.csv") 

# if st.button("Load Data"):
#     df_loaded = None 
#     if file_path == "Iris.csv":
#         try:
#             st.info("No custom file path entered. A **realistic dummy 'Iris.csv'** has been generated for demonstration purposes. Please note that for true high accuracy, an actual Iris dataset is recommended.")
#             df_loaded = generate_realistic_dummy_iris()
#         except Exception as e:
#             st.warning(f"Could not create dummy Iris.csv: {e}")
#     else:
#         try:
#             df_loaded = load_data(file_path)
#         except FileNotFoundError:
#             st.error(f"❌ Error: File not found at '{file_path}'. Please check the path.")
#         except Exception as e:
#             st.error(f"❌ An error occurred during data loading: {e}")

#     if df_loaded is not None:
#         if df_loaded.empty:
#             st.warning("The dataset loaded is empty after cleaning. Please check your CSV file.")
#         elif "Species" not in df_loaded.columns:
#             st.error("The 'Species' column was not found in the dataset. Please ensure your dataset has a 'Species' column (case-sensitive).")
#         else:
#             X = df_loaded.drop("Species", axis=1)
#             y = df_loaded["Species"]
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, stratify=y, random_state=42)
#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(X_train)
#             X_test_scaled = scaler.transform(X_test)
#             X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#             X_test_processed = pd.DataFrame(X_test_scaled, columns=X_test.columns)

#             st.session_state.df = df_loaded
#             st.session_state.X_train = X_train_processed
#             st.session_state.X_test = X_test_processed
#             st.session_state.y_train = y_train
#             st.session_state.y_test = y_test
#             st.session_state.scaler = scaler # Store scaler for future predictions

#             st.success(f"✅ Dataset Loaded, Cleaned, Scaled, and Split 80/20. Total samples: {len(df_loaded)}")
#             st.subheader("First 5 rows of the processed dataset:")
#             st.dataframe(df_loaded.head())
#             st.subheader("Distribution of Species:")
#             st.bar_chart(df_loaded['Species'].value_counts())

# # --- Step 2: Train Model ---
# if st.session_state.X_train is not None:
#     st.header("2. Train Model")
#     st.write("Configure Decision Tree Hyperparameters (Optional):")

#     max_depth = st.slider("Max Depth of Tree", min_value=1, max_value=15, value=5, help="The maximum depth of the tree. A value around 3-7 is often good for Iris to balance bias and variance.")
#     min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, help="The minimum number of samples required to split an internal node.")
#     min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, help="The minimum number of samples required to be at a leaf node.")
#     criterion = st.selectbox("Criterion", ("gini", "entropy"), help="The function to measure the quality of a split.")

#     if st.button("Train Model"):
#         # Define the Decision Tree Classifier with user-selected hyperparameters
#         dt_classifier = DecisionTreeClassifier(
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             criterion=criterion,
#             random_state=42
#         )
        
#         # Create a pipeline that includes scaling and the classifier
#         # This pipeline will be used for cross-validation to ensure proper scaling within each fold
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()), # Scaler applied within each CV fold
#             ('classifier', dt_classifier)
#         ])

#         # Fit the simple classifier on the already scaled training data (for direct prediction metrics)
#         dt_classifier.fit(st.session_state.X_train, st.session_state.y_train)
#         st.session_state.model = dt_classifier # Store the fitted classifier

#         # Store the pipeline for cross-validation
#         st.session_state.pipeline_model = pipeline # Store the pipeline itself

#         st.success("🎯 Model Trained with selected hyperparameters!")

#         st.subheader("Decision Tree Visualization:")
#         fig, ax = plt.subplots(figsize=(20, 10))
#         plot_tree(st.session_state.model, feature_names=st.session_state.X_train.columns, class_names=st.session_state.model.classes_, filled=True, rounded=True, ax=ax)
#         st.pyplot(fig)
#         st.caption("A visualization of the trained Decision Tree. Each node shows the decision rule and class distribution.")

# # --- Step 3: Evaluation ---
# if st.session_state.model is not None:
#     st.header("3. Model Evaluation")
#     st.write("This section provides a comprehensive evaluation of the trained model.")

#     model = st.session_state.model # The classifier fitted on X_train_scaled
#     X_train = st.session_state.X_train
#     X_test = st.session_state.X_test
#     y_train = st.session_state.y_train
#     y_test = st.session_state.y_test

#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)

#     # Calculate metrics
#     train_acc = accuracy_score(y_train, y_train_pred)
#     test_acc = accuracy_score(y_test, y_test_pred)
#     precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
#     recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
#     f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

#     col1, col2, col3 = st.columns(3)
#     col1.metric("✅ Train Accuracy", f"{train_acc*100:.2f}%")
#     col2.metric("📊 Test Accuracy", f"{test_acc*100:.2f}%")
#     col3.metric("🎯 Precision", f"{precision*100:.2f}%")

#     col4, col5, col6 = st.columns(3)
#     col4.metric("📈 Recall", f"{recall*100:.2f}%")
#     col5.metric("🏆 F1 Score", f"{f1*100:.2f}%")
#     col6.metric("🌳 Tree Depth", model.get_depth())

#     st.subheader("Confusion Matrix")
#     cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
#     fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=model.classes_, yticklabels=model.classes_, ax=ax_cm)
#     ax_cm.set_xlabel("Predicted")
#     ax_cm.set_ylabel("Actual")
#     ax_cm.set_title("Confusion Matrix (Test Set)")
#     st.pyplot(fig_cm)
#     st.caption("This matrix shows the number of correct and incorrect predictions made by the classification model compared to the actual outcomes. Diagonal values represent correct predictions.")

#     st.subheader("📉 Generalization Plot")
#     fig_gen, ax_gen = plt.subplots(figsize=(8, 5))
#     metrics_df = pd.DataFrame({
#         'Metric': ['Train Accuracy', 'Test Accuracy', 'Generalization Gap'],
#         'Value': [train_acc, test_acc, abs(train_acc - test_acc)]
#     })
#     sns.barplot(x='Metric', y='Value', data=metrics_df, palette=['green', 'orange', 'red'], ax=ax_gen)
#     ax_gen.set_ylim(0, 1.05)
#     ax_gen.set_ylabel("Accuracy/Difference")
#     ax_gen.set_title("Train vs Test Accuracy & Generalization Gap")
#     st.pyplot(fig_gen)
#     st.caption("The generalization plot visualizes how well the model performs on unseen data compared to the training data. A large 'Generalization Gap' suggests overfitting.")

#     st.subheader("Cross-Validation Scores")
#     st.write("Cross-validation helps in getting a more reliable estimate of model performance and detecting overfitting.")
#     try:
#         if st.session_state.pipeline_model is not None:
#             # Use the original X and y for cross_val_score with the pipeline
#             # The pipeline handles scaling internally for each fold
#             X_original = st.session_state.df.drop("Species", axis=1)
#             y_original = st.session_state.df["Species"]
            
#             # Ensure X_original has consistent column names as expected by the pipeline's scaler
#             X_original.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

#             cv = KFold(n_splits=5, shuffle=True, random_state=42)
#             cv_scores = cross_val_score(st.session_state.pipeline_model, X_original, y_original, cv=cv, scoring='accuracy')

#             st.info(f"Mean Cross-Validation Accuracy: **{np.mean(cv_scores)*100:.2f}%** (Std Dev: {np.std(cv_scores)*100:.2f}%)")
#             st.write("Individual Cross-Validation Scores:")
#             st.write(cv_scores)
#         else:
#             st.warning("Pipeline model not available for cross-validation. Please train the model first.")
#     except Exception as e:
#         st.error(f"Error during cross-validation: {e}")

# # --- Step 4: Prediction ---
# # Ensure both model and scaler are available before allowing prediction
# if st.session_state.model is not None and st.session_state.scaler is not None:
#     st.header("4. Predict New Sample")
#     st.write("Enter the measurements for a new Iris flower to predict its species.")

#     col1, col2 = st.columns(2)
#     with col1:
#         sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, step=0.1, format="%.1f")
#         sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, step=0.1, format="%.1f")
#     with col2:
#         petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, step=0.1, format="%.1f")
#         petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1, format="%.1f")

#     if st.button("Predict Species"):
#         try:
#             # Create a DataFrame for the single sample
#             sample_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
#                                   columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']) # Explicitly set column names
#             # Scale the new sample using the *trained* scaler
#             sample_scaled = st.session_state.scaler.transform(sample_df)
#             prediction = st.session_state.model.predict(sample_scaled)
#             prediction_proba = st.session_state.model.predict_proba(sample_scaled)

#             st.success(f"🔍 Predicted Species: **{prediction[0]}**")
#             st.subheader("Prediction Probabilities:")
#             proba_df = pd.DataFrame(prediction_proba, columns=st.session_state.model.classes_)
#             st.dataframe(proba_df.T.rename(columns={0: 'Probability'}).sort_values(by='Probability', ascending=False))

#         except Exception as e:
#             st.error(f"Error during prediction: {e}")

# st.markdown("---")
# st.caption("Developed with ❤️ using Streamlit and Scikit-learn.")


























import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline 

st.set_page_config(page_title="🌸 Iris Classifier", layout="centered")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'pipeline_model' not in st.session_state:
    st.session_state.pipeline_model = None

@st.cache_data
def load_data(path):
    """
    Loads data from a CSV file, drops 'Id' column if present, and removes duplicates.
    """
    df = pd.read_csv(path)
    df.drop(columns=["Id"], errors='ignore', inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def generate_realistic_dummy_iris():
    """
    Generates a realistic dummy Iris dataset with slightly increased variance
    to encourage deeper decision trees.
    """
    data = []
    # Iris-setosa (50 samples) with slightly increased variance
    for _ in range(50):
        data.append([
            np.random.normal(5.0, 0.3), # SepalLengthCm
            np.random.normal(3.4, 0.3), # SepalWidthCm
            np.random.normal(1.4, 0.2), # PetalLengthCm
            np.random.normal(0.2, 0.1), # PetalWidthCm
            'Iris-setosa'
        ])
    for _ in range(50):
        data.append([
            np.random.normal(5.9, 0.4), # SepalLengthCm
            np.random.normal(2.7, 0.3), # SepalWidthCm
            np.random.normal(4.2, 0.4), # PetalLengthCm
            np.random.normal(1.3, 0.2), # PetalWidthCm
            'Iris-versicolor'
        ])
    # Iris-virginica (50 samples) with slightly increased variance
    for _ in range(50):
        data.append([
            np.random.normal(6.5, 0.5), # SepalLengthCm
            np.random.normal(3.0, 0.3), # SepalWidthCm
            np.random.normal(5.5, 0.5), # PetalLengthCm
            np.random.normal(2.0, 0.2), # PetalWidthCm
            'Iris-virginica'
        ])
    df = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
    # Add a few more samples to slightly increase data points, but not duplicates
    df = pd.concat([df, df.sample(5, random_state=42).reset_index(drop=True)], ignore_index=True)
    df.to_csv("Iris.csv", index=False) # Save the generated dummy data
    return df
st.title("🌸 Iris Flower Classification using Decision Tree")
st.markdown("""
This application classifies Iris flowers into their respective species (Setosa, Versicolor, Virginica)
using a Decision Tree model. Follow the steps below:
1. Load the Iris dataset.
2. Train the Decision Tree model.
3. Evaluate the model's performance.
4. Predict the species of a new Iris flower sample.
""")
st.header("1. Load Dataset")
st.markdown("""
**⭐ IMPORTANT: For best results and high accuracy, please use the actual Iris dataset.**
You can download it from many sources, for example:
- Directly from scikit-learn (programmatically): `from sklearn.datasets import load_iris; iris_data = load_iris(as_frame=True); df = iris_data.frame; df.rename(columns={'sepal length (cm)': 'SepalLengthCm', 'sepal width (cm)': 'SepalWidthCm', 'petal length (cm)': 'PetalLengthCm', 'petal width (cm)': 'PetalWidthCm', 'target': 'Species'}, inplace=True); df['Species'] = df['Species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})`
""")
file_path = st.text_input("Enter file path (e.g., Iris.csv)", value="Iris.csv") 
if st.button("Load Data"):
    df_loaded = None 
    if file_path == "Iris.csv":
        try:
            st.info("No custom file path entered. A **realistic dummy 'Iris.csv'** has been generated for demonstration purposes. Please note that for true high accuracy, an actual Iris dataset is recommended.")
            df_loaded = generate_realistic_dummy_iris()
        except Exception as e:
            st.warning(f"Could not create dummy Iris.csv: {e}")
    else:
        try:
            df_loaded = load_data(file_path)
        except FileNotFoundError:
            st.error(f"❌ Error: File not found at '{file_path}'. Please check the path.")
        except Exception as e:
            st.error(f"❌ An error occurred during data loading: {e}")
    if df_loaded is not None:
        if df_loaded.empty:
            st.warning("The dataset loaded is empty after cleaning. Please check your CSV file.")
        elif "Species" not in df_loaded.columns:
            st.error("The 'Species' column was not found in the dataset. Please ensure your dataset has a 'Species' column (case-sensitive).")
        else:
            numeric_cols = df_loaded.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.error("No numeric features found in the dataset for training. Please ensure your dataset contains numerical measurements.")
            else:
                X = df_loaded[numeric_cols] # Use only numeric columns as features
                y = df_loaded["Species"]
                if y.nunique() < 2 or (y.value_counts() < 2).any():
                    st.warning("Not enough samples per class for stratified split. Falling back to non-stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                X_test_processed = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                st.session_state.df = df_loaded
                st.session_state.X_train = X_train_processed
                st.session_state.X_test = X_test_processed
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler 

                st.success(f"✅ Dataset Loaded, Cleaned, Scaled, and Split 80/20. Total samples: {len(df_loaded)}")
                st.subheader("First 5 rows of the processed dataset:")
                st.dataframe(df_loaded.head())
                st.subheader("Distribution of Species:")
                st.bar_chart(df_loaded['Species'].value_counts())

# --- Step 2: Train Model ---
if st.session_state.X_train is not None:
    st.header("2. Train Model")
    st.write("Configure Decision Tree Hyperparameters (Optional):")

    # Ensure max_depth slider default is appropriate for deeper trees
    max_depth = st.slider("Max Depth of Tree", min_value=1, max_value=15, value=5, help="The maximum depth of the tree. A value around 3-7 is often good for Iris to balance bias and variance.")
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, help="The minimum number of samples required to split an internal node.")
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, help="The minimum number of samples required to be at a leaf node.")
    criterion = st.selectbox("Criterion", ("gini", "entropy"), help="The function to measure the quality of a split.")

    if st.button("Train Model"):
        # Define the Decision Tree Classifier with user-selected hyperparameters
        dt_classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        
        # Create a pipeline that includes scaling and the classifier
        # This pipeline will be used for cross-validation to ensure proper scaling within each fold
        pipeline = Pipeline([
            ('scaler', StandardScaler()), # Scaler applied within each CV fold
            ('classifier', dt_classifier)
        ])

        # Fit the simple classifier on the already scaled training data (for direct prediction metrics)
        dt_classifier.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = dt_classifier # Store the fitted classifier

        # Store the pipeline for cross-validation
        st.session_state.pipeline_model = pipeline # Store the pipeline itself

        st.success("🎯 Model Trained with selected hyperparameters!")

        st.subheader("Decision Tree Visualization:")
        fig, ax = plt.subplots(figsize=(20, 10))
        # Plot the tree. The actual depth shown depends on the trained model's depth,
        # which is influenced by max_depth, min_samples_split, min_samples_leaf, and data complexity.
        plot_tree(st.session_state.model, feature_names=st.session_state.X_train.columns, 
                  class_names=st.session_state.model.classes_, filled=True, rounded=True, ax=ax)
        st.pyplot(fig)
        st.caption("A visualization of the trained Decision Tree. Each node shows the decision rule and class distribution. The actual depth shown depends on the data's complexity and the stopping criteria (like Max Depth).")

# --- Step 3: Evaluation ---
if st.session_state.model is not None:
    st.header("3. Model Evaluation")
    st.write("This section provides a comprehensive evaluation of the trained model.")

    model = st.session_state.model # The classifier fitted on X_train_scaled
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Train Accuracy", f"{train_acc*100:.2f}%")
    col2.metric("📊 Test Accuracy", f"{test_acc*100:.2f}%")
    col3.metric("🎯 Precision", f"{precision*100:.2f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("📈 Recall", f"{recall*100:.2f}%")
    col5.metric("🏆 F1 Score", f"{f1*100:.2f}%")
    col6.metric("🌳 Tree Depth", model.get_depth())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix (Test Set)")
    st.pyplot(fig_cm)
    st.caption("This matrix shows the number of correct and incorrect predictions made by the classification model compared to the actual outcomes. Diagonal values represent correct predictions.")

    st.subheader("📉 Generalization Plot")
    fig_gen, ax_gen = plt.subplots(figsize=(8, 5))
    metrics_df = pd.DataFrame({
        'Metric': ['Train Accuracy', 'Test Accuracy', 'Generalization Gap'],
        'Value': [train_acc, test_acc, abs(train_acc - test_acc)]
    })
    sns.barplot(x='Metric', y='Value', data=metrics_df, palette=['green', 'orange', 'red'], ax=ax_gen)
    ax_gen.set_ylim(0, 1.05)
    ax_gen.set_ylabel("Accuracy/Difference")
    ax_gen.set_title("Train vs Test Accuracy & Generalization Gap")
    st.pyplot(fig_gen)
    st.caption("The generalization plot visualizes how well the model performs on unseen data compared to the training data. A large 'Generalization Gap' suggests overfitting.")

    st.subheader("Cross-Validation Scores")
    st.write("Cross-validation helps in getting a more reliable estimate of model performance and detecting overfitting.")
    try:
        if st.session_state.pipeline_model is not None:
            # Use the original X and y for cross_val_score with the pipeline
            # The pipeline handles scaling internally for each fold
            X_original = st.session_state.df.drop("Species", axis=1)
            y_original = st.session_state.df["Species"]
            
            # Ensure X_original has consistent column names as expected by the pipeline's scaler
            # This is important if original df had non-numeric columns dropped for X_train/test
            if not X_original.empty: # Check if X_original is not empty before assigning columns
                X_original.columns = st.session_state.X_train.columns # Align columns with trained features

                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(st.session_state.pipeline_model, X_original, y_original, cv=cv, scoring='accuracy')

                st.info(f"Mean Cross-Validation Accuracy: **{np.mean(cv_scores)*100:.2f}%** (Std Dev: {np.std(cv_scores)*100:.2f}%)")
                st.write("Individual Cross-Validation Scores:")
                st.write(cv_scores)
            else:
                st.warning("Original dataset (X_original) is empty, cannot perform cross-validation.")
        else:
            st.warning("Pipeline model not available for cross-validation. Please train the model first.")
    except Exception as e:
        st.error(f"Error during cross-validation: {e}")

# --- Step 4: Prediction ---
# Ensure both model and scaler are available before allowing prediction
if st.session_state.model is not None and st.session_state.scaler is not None:
    st.header("4. Predict New Sample")
    st.write("Enter the measurements for a new Iris flower to predict its species.")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, step=0.1, format="%.1f")
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, step=0.1, format="%.1f")
    with col2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, step=0.1, format="%.1f")
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1, format="%.1f")

    if st.button("Predict Species"):
        try:
            # Create a DataFrame for the single sample
            sample_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                     columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']) # Explicitly set column names
            # Scale the new sample using the *trained* scaler
            sample_scaled = st.session_state.scaler.transform(sample_df)
            prediction = st.session_state.model.predict(sample_scaled)
            prediction_proba = st.session_state.model.predict_proba(sample_scaled)

            st.success(f"🔍 Predicted Species: **{prediction[0]}**")
            st.subheader("Prediction Probabilities:")
            proba_df = pd.DataFrame(prediction_proba, columns=st.session_state.model.classes_)
            st.dataframe(proba_df.T.rename(columns={0: 'Probability'}).sort_values(by='Probability', ascending=False))

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-learn.")



