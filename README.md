# KnowledgeOfficer_Assignment

# Step 1: Importing Libraries
The necessary libraries and functions used in the code are imported:
- Libraries: Pandas and Numpy
- Functions: shuffle, CountVectorizer, TfidfTransformer and cross_val_score from sklearn
- Models: MultinomialNB and LinearSVC ; one for each code

# Step 2: Dateset Upload
Uploading the .json dataset using pandas and exploring the data.

# Step 3: Data Exploration
Balancing the Dataset for a fair model training; to avoid biasing towards a specific labeled category

# Step 4: Train-Test Split
Splitting the Data into a train set, used to train the model, and a test set, used to test the model's performance and accuracy before deployment.

# Step 5: Data Preprocessing
Transforming the data into the form required for it to be compatible with the chosen model's input criteria.
- CountVectorizer is used to transform the data into vectors that are later on transformed into arrays for it to be used as train/test sets in the model.
- Tfidf is used to assign the weights to our words, or features, for accurate analysis and predictions

# Step 6: Training and Testing the Model
The model is trained using the training set, created and preprocessed earlier, and then used to predict results for new input, the test set, and the predictions are compared to the actual labels to calculate accuracy

# Step 7: Validation
Cross-Validation is used in this scenario to estimate the model's performance on unseen data and avoid chances of over-fitting the model on the train set.
