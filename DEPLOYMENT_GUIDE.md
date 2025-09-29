# NASA Exoplanet Detection System - Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- macOS/Linux/Windows
- Internet connection for package downloads

### Installation Steps

1. **Clone or download the project files**
   ```bash
   # Navigate to project directory
   cd "Nasa space app 1st"
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv exoplanet_env
   source exoplanet_env/bin/activate  # On Windows: exoplanet_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **For macOS users with XGBoost issues**
   ```bash
   brew install libomp
   ```

5. **Run the system**
   ```bash
   python run_system.py
   ```

6. **Access the web interface**
   - Open your browser and go to: `http://localhost:8050`

## 📁 Project Structure

```
Nasa space app 1st/
├── cumulative_2025.09.21_20.01.51.csv     # Kepler dataset
├── k2pandc_2025.09.21_20.02.10.csv        # K2 dataset  
├── TOI_2025.09.21_20.02.02.csv            # TESS TOI dataset
├── exoplanet_ml_pipeline.py               # ML training pipeline
├── exoplanet_web_app.py                   # Web interface
├── run_system.py                          # System launcher
├── requirements.txt                       # Python dependencies
├── DEPLOYMENT_GUIDE.md                    # This file
├── exoplanet_env/                         # Virtual environment
├── best_exoplanet_model.pkl              # Trained model (generated)
├── feature_scaler.pkl                    # Feature scaler (generated)
└── label_encoder.pkl                     # Label encoder (generated)
```

## 🔧 System Components

### 1. Machine Learning Pipeline (`exoplanet_ml_pipeline.py`)
- **Purpose**: Train ML models on NASA exoplanet datasets
- **Features**:
  - Processes Kepler, K2, and TESS data
  - Feature engineering and data preprocessing
  - Multiple ML algorithms (XGBoost, Random Forest, Neural Networks)
  - Model evaluation and selection
  - Handles class imbalance

### 2. Web Interface (`exoplanet_web_app.py`)
- **Purpose**: Interactive web application for exoplanet classification
- **Features**:
  - Real-time classification of new candidates
  - Data visualization and exploration
  - Model performance metrics
  - Educational content about exoplanets

### 3. System Launcher (`run_system.py`)
- **Purpose**: Simplified startup script
- **Features**:
  - Automatic model training if needed
  - Error handling and status reporting
  - One-command system launch

## 🎯 Using the Web Interface

### Classification Tab
1. Enter exoplanet parameters:
   - **Orbital Parameters**: Period, duration
   - **Transit Parameters**: Depth, planet radius
   - **Physical Parameters**: Temperature, insolation
   - **Stellar Parameters**: Temperature, gravity, radius, magnitude

2. Click "Classify Exoplanet" to get:
   - Classification result (CONFIRMED/CANDIDATE/FALSE_POSITIVE)
   - Confidence percentage
   - Probability breakdown for each class

### Visualization Tab
- **Scatter Plots**: Explore relationships between features
- **Class Distribution**: See proportion of each exoplanet type
- **Correlation Heatmap**: Understand feature relationships

### Performance Tab
- **Model Metrics**: Accuracy, ROC AUC, precision, recall
- **Feature Importance**: Which parameters matter most
- **Model Comparison**: Performance across different algorithms

## 📊 Model Performance

### Best Model: XGBoost Classifier
- **Accuracy**: 79.38%
- **ROC AUC**: 92.34%
- **Precision**: 79% (macro average)
- **Recall**: 79% (macro average)

### Key Features for Classification
1. Transit depth and duration
2. Orbital period
3. Planet radius
4. Stellar parameters (temperature, radius, gravity)
5. Signal strength metrics

## 🔬 Technical Details

### Data Processing
- **Datasets Combined**: ~16,675 objects from 3 NASA missions
- **Feature Engineering**: 17 features including derived parameters
- **Class Balancing**: Undersampling to handle imbalanced classes
- **Preprocessing**: Standardization, missing value imputation

### Machine Learning Approach
- **Algorithms Tested**: 6 different ML models
- **Validation**: Train/test split with stratification
- **Metrics**: Multi-class classification with weighted averages
- **Feature Scaling**: StandardScaler for algorithm compatibility

## 🌐 Deployment Options

### Local Development
```bash
python run_system.py
# Access at http://localhost:8050
```

### Production Deployment
For production deployment, consider:

1. **Docker Containerization**
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 8050
   CMD ["python", "exoplanet_web_app.py"]
   ```

2. **Cloud Platforms**
   - **Heroku**: Use `Procfile` with gunicorn
   - **AWS**: Deploy on EC2 or ECS
   - **Google Cloud**: Use App Engine or Cloud Run

3. **Environment Variables**
   ```bash
   export DASH_HOST=0.0.0.0
   export DASH_PORT=8050
   export DASH_DEBUG=False
   ```

## 🐛 Troubleshooting

### Common Issues

1. **XGBoost Import Error on macOS**
   ```bash
   brew install libomp
   ```

2. **Permission Errors**
   ```bash
   chmod +x run_system.py
   ```

3. **Port Already in Use**
   - Change port in `exoplanet_web_app.py` (line with `port=8050`)
   - Or kill existing process: `lsof -ti:8050 | xargs kill`

4. **Memory Issues with Large Datasets**
   - Reduce dataset size in pipeline
   - Increase system RAM or use cloud instance

5. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Performance Optimization

1. **Faster Training**
   - Reduce `n_estimators` in Random Forest
   - Use fewer cross-validation folds
   - Sample smaller subset of data

2. **Web App Performance**
   - Enable caching for visualizations
   - Use plotly's `figure_factory` for complex plots
   - Implement lazy loading for large datasets

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages in terminal
3. Ensure all dependencies are correctly installed
4. Verify Python version compatibility (3.8+)

## 🎓 Educational Use

This system is designed for:
- **Researchers**: Classify new exoplanet candidates
- **Students**: Learn about exoplanet detection methods
- **Educators**: Demonstrate machine learning in astronomy
- **General Public**: Explore NASA's exoplanet discoveries

## 📈 Future Enhancements

Potential improvements:
- Real-time data ingestion from NASA APIs
- Advanced neural network architectures
- Time-series analysis for light curves
- Integration with astronomical databases
- Multi-wavelength data analysis
- Automated follow-up observation scheduling

---

**Built for NASA Space Apps Challenge 2025** 🌟
**Advancing exoplanet discovery through AI and machine learning** 🔭
