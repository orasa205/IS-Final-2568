import streamlit as st

st.set_page_config(page_title="Reference", page_icon="📚", layout="wide")

st.markdown("""
<style>
    h1 { color: #2c3e50; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    .ref-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .kaggle-box {
        background: linear-gradient(135deg, #20beff 0%, #43afe9 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    .github-box {
        background: linear-gradient(135deg, #24292e 0%, #2f363d 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 References & Sources")
st.markdown("---")

st.header("📖 Dataset Sources")

st.markdown("""
<div class="kaggle-box">
    <h3 style="color: white; margin-top: 0;">📊 Kaggle Datasets</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="ref-box">
        <h4 style="margin-top: 0;">🌸 Iris Dataset</h4>
        <p><b>Dataset Name:</b> Iris Species</p>
        <p><b>Source:</b> Kaggle</p>
        <p><b>URL:</b> <a href="https://www.kaggle.com/datasets" target="_blank">kaggle.com/datasets</a></p>
        <hr>
        <p><b>Citation:</b></p>
        <blockquote style="font-style: italic; color: #555;">
            Fisher, R.A. (1988). Iris [Data set]. Kaggle.<br>
            Originally published in: Fisher, R.A. (1936) "The use of multiple measurements in taxonomic problems", 
            Annals of Eugenics, 7, Part II, 179-188.
        </blockquote>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="ref-box">
        <h4 style="margin-top: 0;">🎓 Student Performance Dataset</h4>
        <p><b>Dataset Name:</b> Student Performance Factors</p>
        <p><b>Source:</b> Kaggle</p>
        <p><b>URL:</b> <a href="https://www.kaggle.com/datasets" target="_blank">kaggle.com/datasets</a></p>
        <hr>
        <p><b>Citation:</b></p>
        <blockquote style="font-style: italic; color: #555;">
            Author. (Year). Student Performance Factors Dataset. Kaggle.<br>
            Retrieved from https://www.kaggle.com/datasets
        </blockquote>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("📚 Algorithm References")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="ref-box">
        <h4 style="margin-top: 0;">🔢 Machine Learning Algorithms</h4>
        <ul style="line-height: 1.8;">
            <li><b>Linear Regression:</b> James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning. Springer.</li>
            <li><b>Random Forest:</b> Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.</li>
            <li><b>K-Nearest Neighbors:</b> Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory.</li>
            <li><b>Voting Regressor:</b> Dietterich, T.G. (2000). Ensemble methods in machine learning. Multiple Classifier Systems.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="ref-box">
        <h4 style="margin-top: 0;">🧠 Neural Network & Deep Learning</h4>
        <ul style="line-height: 1.8;">
            <li><b>Artificial Neural Networks:</b> Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review.</li>
            <li><b>ReLU Activation:</b> Nair, V., & Hinton, G.E. (2010). Rectified linear units improve restricted boltzmann machines. ICML.</li>
            <li><b>Softmax:</b> Bridle, J.S. (1990). Probabilistic interpretation of feedforward classification network outputs. Neurocomputing.</li>
            <li><b>Backpropagation:</b> Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. Nature.</li>
            <li><b>Adam Optimizer:</b> Kingma, D.P., & Ba, J. (2014). Adam: A method for stochastic optimization. ICLR.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("📦 Libraries & Frameworks")

st.markdown("""
<div class="ref-box">
    <table style="width: 100%;">
        <tr style="background: #f8f9fa;">
            <th style="padding: 10px; text-align: left;">Library</th>
            <th style="padding: 10px; text-align: left;">Version</th>
            <th style="padding: 10px; text-align: left;">Purpose</th>
        </tr>
        <tr>
            <td style="padding: 10px;">Streamlit</td>
            <td style="padding: 10px;">Latest</td>
            <td style="padding: 10px;">Web Application Framework</td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 10px;">Pandas</td>
            <td style="padding: 10px;">Latest</td>
            <td style="padding: 10px;">Data Manipulation</td>
        </tr>
        <tr>
            <td style="padding: 10px;">NumPy</td>
            <td style="padding: 10px;">Latest</td>
            <td style="padding: 10px;">Numerical Computing</td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 10px;">Scikit-learn</td>
            <td style="padding: 10px;">Latest</td>
            <td style="padding: 10px;">Machine Learning</td>
        </tr>
        <tr>
            <td style="padding: 10px;">TensorFlow</td>
            <td style="padding: 10px;">Latest</td>
            <td style="padding: 10px;">Deep Learning</td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 10px;">Keras</td>
            <td style="padding: 10px;">Latest</td>
            <td style="padding: 10px;">Neural Networks API</td>
        </tr>
        <tr>
            <td style="padding: 10px;">Matplotlib</td>
            <td style="padding: 10px;">Latest</td>
            <td style="padding: 10px;">Data Visualization</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.header("📖 Academic References")

st.markdown("""
<div class="ref-box">
    <h4 style="margin-top: 0;">📚 Related Papers & Books</h4>
    <ol style="line-height: 2;">
        <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.</li>
        <li>Bishop, C.M. (2006). Pattern Recognition and Machine Learning. Springer.</li>
        <li>Haykin, S. (2009). Neural Networks and Learning Machines. Pearson.</li>
        <li>Zhang, A., Lipton, Z.C., Li, M., & Smola, A.J. (2021). Dive into Deep Learning. Apache MXNet.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.header("🌐 Online Resources")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="ref-box">
        <h4 style="margin-top: 0;">🔗 Useful Links</h4>
        <ul style="line-height: 1.8;">
            <li><a href="https://scikit-learn.org/" target="_blank">Scikit-learn Documentation</a></li>
            <li><a href="https://www.tensorflow.org/api_docs" target="_blank">TensorFlow Documentation</a></li>
            <li><a href="https://keras.io/api/" target="_blank">Keras Documentation</a></li>
            <li><a href="https://streamlit.io/library" target="_blank">Streamlit Documentation</a></li>
            <li><a href="https://matplotlib.org/stable/" target="_blank">Matplotlib Documentation</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="ref-box">
        <h4 style="margin-top: 0;">📖 Tutorials</h4>
        <ul style="line-height: 1.8;">
            <li><a href="https://www.kaggle.com/learn" target="_blank">Kaggle Learn</a></li>
            <li><a href="https://www.coursera.org/" target="_blank">Coursera ML Courses</a></li>
            <li><a href="https://www.deeplearning.ai/" target="_blank">DeepLearning.AI</a></li>
            <li><a href="https://www.tensorflow.org/tutorials" target="_blank">TensorFlow Tutorials</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; color: white; text-align: center;">
    <h3 style="color: white;">📝 Citation</h3>
    <p style="font-size: 16px;">
        If you use this application or reference the datasets, please cite appropriately.
    </p>
    <p style="font-style: italic;">
        Built with ❤️ using Streamlit, Scikit-learn, and TensorFlow
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.info("📚 Navigate to other pages using the sidebar!")
