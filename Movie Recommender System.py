import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re

# تنظیمات صفحه
st.set_page_config(
    page_title="سیستم پیشنهاد فیلم",
    page_icon="🎬",
    layout="wide"
)

# استایل فارسی
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-right: 4px solid #1f77b4;
    }
    .similarity-score {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """لود داده‌ها با کش"""
    try:
        data = pd.read_csv(r"C:\Users\User\Desktop\my project\3  (9.11.2025)\results_with_crew.csv")
        
        # پیش‌پردازش داده‌ها
        data.drop(["tconst", "numVotes", "Title_IMDb_Link"], axis=1, inplace=True, errors='ignore')
        
        data = data.rename(columns={
            'primaryTitle': 'movie_name',
            'startYear': 'year',
        })
        
        data["writers"] = data["writers"].fillna("unknown_writer")
        data['year'] = data['year'].astype(str).str.extract(r'(\d{4})')[0]
        
        data["genres"] = data["genres"].str.replace(",", " ")
        data["writers"] = data["writers"].str.replace(",", " ")
        data["directors"] = data["directors"].str.replace(",", " ")
        
        data["combined_feature"] = (
            data["genres"] + " " + 
            data["directors"] + " " + 
            data["writers"] + " " +
            data["year"].astype(str) + " " +
            "rating_" + data["averageRating"].round(1).astype(str) + " " +
            "runtime_" + data["runtimeMinutes"].astype(str)
        )
        
        return data
    except Exception as e:
        st.error(f"خطا در لود داده‌ها: {e}")
        return None

@st.cache_resource
def create_similarity_matrix(_data):
    """ایجاد ماتریس شباهت با کش"""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        vector = vectorizer.fit_transform(_data["combined_feature"])
        similarities = cosine_similarity(vector, dense_output=False)
        return similarities
    except Exception as e:
        st.error(f"خطا در ایجاد ماتریس شباهت: {e}")
        return None

def recommend_movie(movie_title, data, similarities, k=5, min_rating=7.0):
    """تابع پیشنهاد فیلم"""
    try:
        # پیدا کردن فیلم
        movie_idx = data[data["movie_name"].str.strip().str.lower() == movie_title.strip().lower()].index[0]
        
        # محاسبه شباهت‌ها
        similarities_score = list(enumerate(similarities[movie_idx].toarray().flatten()))
        sorted_scores = sorted(similarities_score, key=lambda x: x[1], reverse=True)
        
        # فیلتر و انتخاب
        recommendations = []
        for idx, score in sorted_scores[1:k*2]:
            if len(recommendations) >= k:
                break
            if data.iloc[idx]["averageRating"] >= min_rating and idx != movie_idx:
                movie_data = data.iloc[idx]
                recommendations.append({
                    'title': movie_data["movie_name"],
                    'rating': movie_data["averageRating"],
                    'year': movie_data["year"],
                    'genres': movie_data["genres"],
                    'director': movie_data["directors"].split()[0] if pd.notna(movie_data["directors"]) else "نامشخص",
                    'similarity_score': round(score, 3)
                })
        
        if not recommendations:
            return "هیچ پیشنهادی پیدا نشد. لطفاً فیلترها را تغییر دهید."
        
        return recommendations
        
    except IndexError:
        return "فیلم پیدا نشد. لطفاً عنوان دقیق فیلم را وارد کنید."
    except Exception as e:
        return f"خطا در پردازش: {e}"

def main():
    """تابع اصلی برنامه"""
    
    # هدر اصلی
    st.markdown('<div class="main-header">🎬 سیستم پیشنهاد فیلم</div>', unsafe_allow_html=True)
    st.write("### جستجوی فیلم و دریافت پیشنهاد‌های مشابه")
    
    # لود داده‌ها
    with st.spinner("در حال لود داده‌ها..."):
        data = load_data()
    
    if data is None:
        st.error("امکان لود داده‌ها وجود ندارد. لطفاً مسیر فایل را بررسی کنید.")
        return
    
    # ایجاد ماتریس شباهت
    with st.spinner("در حال محاسبه شباهت‌ها..."):
        similarities = create_similarity_matrix(data)
    
    if similarities is None:
        st.error("امکان ایجاد ماتریس شباهت وجود ندارد.")
        return
    
    # سایدبار برای فیلترها
    with st.sidebar:
        st.header("⚙️ تنظیمات پیشنهاد")
        k = st.slider("تعداد پیشنهادها:", min_value=1, max_value=10, value=5)
        min_rating = st.slider("حداقل امتیاز:", min_value=5.0, max_value=9.0, value=7.0, step=0.1)
        
        st.header("🔍 جستجوی سریع")
        popular_movies = data.head(10)["movie_name"].tolist()
        selected_popular = st.selectbox("فیلم‌های محبوب:", [""] + popular_movies)
    
    # بخش اصلی
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("جستجوی فیلم")
        
        # ورودی جستجو
        if selected_popular:
            movie_title = st.text_input("عنوان فیلم:", value=selected_popular)
        else:
            movie_title = st.text_input("عنوان فیلم:", placeholder="مثال: The Godfather")
        
        # دکمه جستجو
        if st.button("🎯 دریافت پیشنهادها", type="primary", use_container_width=True):
            if movie_title.strip():
                with st.spinner("در حال یافتن فیلم‌های مشابه..."):
                    recommendations = recommend_movie(movie_title, data, similarities, k, min_rating)
                
                if isinstance(recommendations, list):
                    st.success(f"🎉 {len(recommendations)} پیشنهاد برای فیلم '{movie_title}' پیدا شد!")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-box">
                                <h4>#{i} {rec['title']} ({rec['year']})</h4>
                                <p><strong>ژانر:</strong> {rec['genres']}</p>
                                <p><strong>کارگردان:</strong> {rec['director']}</p>
                                <p><strong>امتیاز:</strong> ⭐ {rec['rating']}/10</p>
                                <p><strong>میزان شباهت:</strong> <span class="similarity-score">{rec['similarity_score']}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(recommendations)
            else:
                st.warning("⚠️ لطفاً عنوان فیلم را وارد کنید!")
    
    with col2:
        st.subheader("📊 اطلاعات دیتاست")
        st.metric("تعداد کل فیلم‌ها", f"{len(data):,}")
        st.metric("میانگین امتیاز", f"{data['averageRating'].mean():.2f}")
        st.metric("بازه سال‌ها", f"{data['year'].min()} - {data['year'].max()}")
        
        st.subheader("🎭 ژانرهای برتر")
        top_genres = data['genres'].str.split().explode().value_counts().head(5)
        for genre, count in top_genres.items():
            st.write(f"- {genre}: {count}")

if __name__ == "__main__":
    main()