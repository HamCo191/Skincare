import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

st.set_page_config(page_title="Skin Care Recommender System", page_icon=":rose:", layout="wide")

# Load the dataset
skincare = pd.read_csv("MP-Skin Care Product Recommendation System3.csv", encoding='utf-8')

# Preprocessing: Split 'notable_effects' and 'skintype' into separate binary columns
notable_effects_dummies = skincare['notable_effects'].str.get_dummies(sep=', ')
skintype_dummies = skincare['skintype'].str.get_dummies(sep=', ')

# Concatenate the original dataframe with these new binary columns
skincare_processed = pd.concat([skincare, notable_effects_dummies, skintype_dummies], axis=1)

# Drop original 'notable_effects' and 'skintype' columns
skincare_processed.drop(['notable_effects', 'skintype'], axis=1, inplace=True)

# Select a single notable effect for binary classification
target_effect = 'Brightening'

# Check if the notable effect is in the columns
if target_effect in notable_effects_dummies.columns:
    y = notable_effects_dummies[target_effect]
else:
    y = pd.Series([0] * len(notable_effects_dummies))  # Default to zeros if not found

# Features for model training
features = skincare_processed.drop(['product_href', 'product_name', 'product_type', 'brand', 'price', 'description', 'picture_src'], axis=1)

# Split the features and the new target
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Display accuracy metrics in Streamlit
st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1 Score: {f1}")

# Streamlit UI
EXAMPLE_NO = 2

def streamlit_menu(example=1):
    if example == 1:
        with st.sidebar:
            selected = st.radio(
                "Main Menu", 
                options=["Skin Care", "Get Recommendation", "Skin Care 101"], 
                index=0
            )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    st.title(f"{selected} Product Recommender :sparkles:")
    st.write('---') 
    st.write(
        """
        ##### **Aplikasi Rekomendasi Produk Perawatan Kulit merupakan salah satu implementasi Machine Learning yang dapat memberikan rekomendasi produk perawatan kulit sesuai dengan jenis dan permasalahan kulit Anda**
        """)
    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time = 1) #displaying the video 
    st.write(' ') 
    st.write(' ')
    st.write(
        """
        ##### Anda akan mendapatkan rekomendasi produk skin care dari berbagai macam brand kosmetik dengan total 1200+ produk yang disesuaikan dengan kebutuhan kulit Anda. 
        ##### Terdapat 5 kategori produk skin care dengan 5 tipe kulit berbeda, serta permasalahan dan manfaat yang ingin didapatkan dari produk. Aplikasi rekomendasi ini hanyalah sebuah sistem yang memberikan rekomendasi sesuai dengan data yang Anda masukkan, bukan konsultasi ilmiah.
        ##### Silahkan pilih halaman *Get Recommendation* untuk mulai mendapatkan rekomendasi Atau pilih halaman *Skin Care 101* untuk melihat tips dan trik seputar skin care
        """)
    st.write("**Selamat Mencoba :) !**")
    st.info('Credit: Created by Dwi Ayu Nouvalina')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    st.write(
        """
        ##### **Untuk mendapatkan rekomendasi, silahkan masukkan jenis kulit, keluhan, dan manfaat yang diinginkan untuk mendapatkan rekomendasi produk skin care yang tepat**
        """) 
    st.write('---') 

    first,last = st.columns(2)

    category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique())
    category_pt = skincare[skincare['product_type'] == category]

    skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    prob = st.multiselect(label='Skin Problems : ', options= ['Kulit Kusam', 'Jerawat', 'Bekas Jerawat','Pori-pori Besar', 'Flek Hitam', 'Garis Halus dan Kerutan', 'Komedo', 'Warna Kulit Tidak Merata', 'Kemerahan', 'Kulit Kendur'])

    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    selected_options = st.multiselect('Notable Effects : ',opsi_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    product = st.selectbox(label='Produk yang Direkomendasikan Untuk Kamu', options = sorted(opsi_pn))

    tf = TfidfVectorizer()
    tf.fit(skincare['notable_effects']) 
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 
    cosine_sim = cosine_similarity(tfidf_matrix) 
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):
        index = similarity_data.loc[:,nama_produk].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1:-(k+2):-1]]
        closest = closest.drop(nama_produk, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return
