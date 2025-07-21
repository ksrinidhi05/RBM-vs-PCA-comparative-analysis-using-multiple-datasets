project:
  title: "RBM vs PCA: A Comparative Study of Dimensionality Reduction Techniques"

  overview: >
    This project presents a comparative analysis between Principal Component Analysis (PCA)
    and Restricted Boltzmann Machines (RBM) — two widely used dimensionality reduction techniques.
    The goal is to evaluate their performance in reducing high-dimensional data to a lower-dimensional
    representation while preserving key information, with applications in reconstruction, visualization,
    and classification tasks.

    Dimensionality reduction helps simplify data, reduce noise, improve training time for machine learning
    models, and make visualizations possible. While PCA is a classical linear technique, RBM is a non-linear
    probabilistic neural network capable of modeling more complex feature interactions.

folder_structure:
  - project-root/
  - code/
    - pca.py
    - rbm.py
    - utils.py
    - visualization.py
  - data/
  - results/
  - README.md

techniques:
  - name: "Principal Component Analysis (PCA)"
    type: "Linear, unsupervised learning method"
    goal: "Project data onto directions of maximum variance"
    pros:
      - Fast
      - Interpretable
      - Deterministic
    cons:
      - Cannot capture non-linear relationships

  - name: "Restricted Boltzmann Machine (RBM)"
    type: "Generative stochastic neural network"
    goal: "Learn probability distribution over input features using hidden units"
    pros:
      - Can model non-linear patterns
    cons:
      - Slower to train
      - Sensitive to hyperparameters

dataset:
  name: "MNIST"
  description: "28x28 grayscale images of handwritten digits (0–9)"
  features: 784
  train_samples: 60000
  test_samples: 10000
  note: "Can be replaced with Fashion-MNIST by updating the data loader"

objectives:
  - Implement and apply PCA and RBM
  - Reduce features to 2D and 3D for visualization
  - Visualize class separation
  - Measure reconstruction loss
  - Evaluate classification accuracy
  - Compare performance and representation

usage:
  clone:
    - git clone https://github.com/your-username/your-repo-name.git
    - cd your-repo-name
  install:
    - pip install -r requirements.txt
  run:
    pca: python code/pca.py
    rbm: python code/rbm.py
    visualization: python code/visualization.py

evaluation_metrics:
  - Reconstruction Error
  - Latent Space Visualization (2D and 3D)
  - Classification Accuracy (on reduced features)

results_summary:
  - metric: "Reconstruction Error"
    pca: "Moderate (due to linearity)"
    rbm: "Lower (captures non-linearity)"
  - metric: "Classification Accuracy"
    pca: "~85-90%"
    rbm: "~90-95%"
  - metric: "Training Time"
    pca: "Fast"
    rbm: "Slower (depends on epochs)"
  - metric: "Interpretability"
    pca: "High"
    rbm: "Low"
  - metric: "Visualization Quality"
    pca: "Good for linear separation"
    rbm: "Better class separation"

sample_outputs:
  - name: "2D PCA Projection"
    image: "results/pca_2d.png"
  - name: "2D RBM Projection"
    image: "results/rbm_2d.png"

tools_libraries:
  - Python 3.x
  - NumPy
  - Pandas
  - scikit-learn
  - PyTorch
  - Matplotlib
  - Seaborn

learnings:
  - PCA is simple and efficient but linear
  - RBMs are more complex and capture non-linear features
  - Dimensionality reduction reduces training time and preserves structure

future_improvements:
  - Add Autoencoders and t-SNE to the comparison
  - Extend to more complex datasets like CIFAR-10
  - Explore clustering and anomaly detection
  - Hyperparameter tuning for RBM (hidden units, learning rate)

acknowledgements:
  - "MNIST Dataset: http://yann.lecun.com/exdb/mnist/"
  - "RBM architecture based on work by Hinton and PyTorch examples"

contact:
  email: "your-email@example.com"
  github_issues: "Open an issue in the repository"
