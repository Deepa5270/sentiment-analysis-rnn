# Sentiment Analysis with RNN

![Sentiment Analysis Banner](https://via.placeholder.com/900x300?text=Sentiment+Analysis+using+SimpleRNN+%7C+Deep+Learning+%7C+NLP)

Deep learning model for sentiment classification using Recurrent Neural Networks (RNN) on IMDB movie reviews.

## 🌟 Features

- **Pre-trained SimpleRNN model** with 85%+ accuracy
- **Word embeddings** for efficient text representation  
- **IMDB dataset** (50,000 reviews) training pipeline
- **Streamlit web application** for interactive predictions
- **Jupyter notebooks** for training and analysis
- **Fast inference** (<100ms per prediction)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Deepa5270/sentiment-analysis-rnn.git
cd sentiment-analysis-rnn

# Install dependencies
pip install -r requirements.txt

# Run web application
streamlit run main.py
```

Open your browser at `http://localhost:8501`

## 📊 Model Architecture

![Model Architecture Diagram](https://via.placeholder.com/800x450?text=Input+Text+%E2%86%92+Embedding+%E2%86%92+SimpleRNN+%E2%86%92+Dense+%E2%86%92+Output)

```
Text Input (Max 500 characters)
          ↓
    Embedding Layer
    (32 dimensions)
          ↓
    SimpleRNN Layer
    (128 units)
          ↓
    Dropout (0.5)
          ↓
    Dense Layer
    (64 units)
          ↓
  Output (Sigmoid)
  Positive/Negative
```

## 📈 Performance Metrics

![Performance Chart](https://via.placeholder.com/700x350?text=Accuracy%3A+85.8%25+%7C+Inference%3A+%3C100ms)

| Metric | Value |
|--------|-------|
| Test Accuracy | 85.8% |
| Inference Time | <100ms |
| Model Size | 2.3 MB |
| Training Time | ~5 minutes |
| Vocabulary Size | 10,000 words |
| Max Sequence Length | 500 tokens |

## 💻 Usage

### Via Web Application (Recommended)

```bash
streamlit run main.py
```

Then type your movie review and see instant sentiment prediction!

### Via Python Code

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load pre-trained model
model = load_model('simple_rnn_imdb.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Predict sentiment
text = "This movie was absolutely amazing!"
sequence = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequence, maxlen=500)
prediction = model.predict(padded)

# Results
sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"
confidence = prediction[0][0]

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
```

## 📁 Project Structure

```
sentiment-analysis-rnn/
├── main.py                    # Streamlit web application
├── simple_rnn_imdb.h5        # Pre-trained model weights
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
│
├── embedding.ipynb            # Embedding layer analysis
├── simplernn.ipynb            # Model training notebook
├── prediction.ipynb           # Inference examples
│
└── .gitignore                 # Git configuration
```

## 📚 Notebooks

| Notebook | Purpose | Time |
|----------|---------|------|
| **embedding.ipynb** | Analyze word embeddings | 10 min |
| **simplernn.ipynb** | Train custom models | 20 min |
| **prediction.ipynb** | Test predictions | 15 min |

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- NumPy, Pandas, Scikit-learn

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Information

- **Source**: IMDB Movie Reviews
- **Total Reviews**: 50,000
- **Training Set**: 25,000 reviews
- **Test Set**: 25,000 reviews
- **Labels**: Positive (1) / Negative (0)
- **Vocabulary**: 10,000 most frequent words

## 🎯 Model Training Details

**Training Configuration:**
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Epochs: 10
- Validation Split: 20%
- Early Stopping: Yes (patience: 2)

## 🌐 Live Demo

![App Preview](https://via.placeholder.com/800x500?text=Interactive+Sentiment+Analyzer+App)

**Try the live application:**

👉 [**View Live Demo**](https://sentiment-analysis-rnn-lccbnqgswwv7jtccdergwq.streamlit.app/)

Test your own movie reviews and see instant sentiment predictions!

## 💡 Example Predictions

```
Input: "This movie was fantastic!"
Output: Positive (Confidence: 94%)

Input: "Worst film I've ever seen"
Output: Negative (Confidence: 97%)

Input: "It was okay, nothing special"
Output: Negative (Confidence: 58%)
```

## 📄 License

MIT License - See LICENSE file for details

This means:
- ✅ Free to use commercially
- ✅ Free to modify
- ✅ Free to distribute
- ✅ Must include license notice

## 👨‍💻 Author

**Deepa Sharma**

- GitHub: [@Deepa5270](https://github.com/Deepa5270)
- Project Repository: [sentiment-analysis-rnn](https://github.com/Deepa5270/sentiment-analysis-rnn)

## 🤝 Contributing

Contributions are welcome! If you'd like to:
- Report bugs
- Suggest improvements
- Submit code changes

Please open a [GitHub Issue](https://github.com/Deepa5270/sentiment-analysis-rnn/issues)

## 📞 Support

- 💬 Questions? Open a GitHub Issue
- 🐛 Found a bug? Report it on GitHub Issues
- 💡 Have an idea? Share it in Discussions

## 🙏 Acknowledgments

- TensorFlow and Keras for deep learning framework
- IMDB dataset contributors
- Streamlit for easy web deployment
- Open-source community

---

<div align="center">

**Made with ❤️ by Deepa Sharma**

⭐ If you find this project helpful, consider giving it a star on GitHub!

[View on GitHub](https://github.com/Deepa5270/sentiment-analysis-rnn)

</div>
