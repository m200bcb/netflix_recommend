import io

import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_path = ''
df = pd.read_csv(data_path + 'netflix_titles.csv')
new_df = pd.read_csv(data_path + 'netflix_titles_no_missing_values.csv')

alphabet_titles = {chr(i): new_df[new_df['title'].str.startswith(chr(i))]['title'].tolist() for i in range(65, 91)}

# 세션 상태 초기화
if 'selected_alphabet_1' not in st.session_state:
    st.session_state['selected_alphabet_1'] = 'A'
if 'movie_title_list_1' not in st.session_state:
    st.session_state['movie_title_list_1'] = alphabet_titles[st.session_state['selected_alphabet_1']]

if 'selected_alphabet_2' not in st.session_state:
    st.session_state['selected_alphabet_2'] = 'A'
if 'movie_title_list_2' not in st.session_state:
    st.session_state['movie_title_list_2'] = alphabet_titles[st.session_state['selected_alphabet_2']]

def update_movie_list_1():
    st.session_state['movie_title_list_1'] = alphabet_titles[st.session_state['selected_alphabet_1']]

def update_movie_list_2():
    st.session_state['movie_title_list_2'] = alphabet_titles[st.session_state['selected_alphabet_2']]


options = ['설명', '(1) 결측치 제거', '(2) 시각화', '(3) 영화 기반 추천', '(4) 취향 기반 추천']

with st.sidebar:
    menu = option_menu(
        menu_title='메뉴',
        options=options,
        menu_icon='youtube',
        styles={
            'container': {'padding': '5!important', 'background-color': 'Pink'},
            'icon': {'color': 'white', 'font-size': '18px'},
            'nav-link': {'color': 'white', 'font-size': '18px',
                         'text-align': 'left', 'margin': '0px',
                         '--hover-color': 'LightPink'},
            'nav-link-selected': {'background-color': 'HotPink'}
        }
    )

i = options.index(menu)

if i == 0:
    st.header('Netflix 추천 시스템')
    st.subheader('개요')
    st.markdown('''
    :red[넷플릭스]에서도 제공하고 있는 콘텐츠 추천 시스템은 크게 2가지 방식으로 분류할 수 있습니다.
    첫 번째는 비슷한 콘텐츠를 본 사람들의 평점 데이터를 이용하는 방식이고,
    두 번째는 콘텐츠 자체의 제작 국가, 종류 등의 데이터를 이용해 유사 콘텐츠를 추천하는 방식입니다.
    여기서는 두 번째 방식을 사용합니다.
    ''')

    st.subheader('진행 과정')
    st.markdown('''
    1. 주어진 데이터 정보를 파악한 후 결측치를 제거합니다.  
    2. 주어진 넷플릭스 콘텐츠 데이터에 대해 국가별, 길이별, 종류별 데이터 분석을 실시하고 이를 시각화합니다.  
    3. 코사인 유사도를 이용해 주어진 콘텐츠 제목과 관련된 콘텐츠를 추천합니다.  
    4. 샘플 사용자 데이터가 주어졌을 때 해당 사용자에게 적합한 콘텐츠를 추천하는 과정을 진행합니다.  
    ''')

    st.subheader('목표')
    st.markdown('''
    - 사용자가 자신이 시청한 콘텐츠를 그날그날 별점과 함께 입력하면 지금까지 본 콘텐츠의 별점을 기반으로 해당 사용자에게 맞춤 콘텐츠를 추천해주는 것을 목표로 합니다.  
    - 이 과정에서 사용자의 국가, 길이, 콘텐츠 종류 취향과 줄거리 유사도가 고려되며, 최근에 평가한 콘텐츠에 대해 더 높은 비중을 두고 반영하도록 합니다.
    ''')

if i == 1:
    st.subheader('(1) 결측치 제거')

    st.markdown('- df.info()')
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.markdown('- df.head()')
    st.dataframe(df.head())

    st.markdown('- 결측치 확인')
    st.text(df.isnull().sum())

    st.markdown('- rating에 결측치가 있는 항목 출력')
    st.text(df[df['rating'].isnull()])

    st.markdown('- duration에 결측치가 있는 항목 출력')
    st.text(df[df['duration'].isnull()])
    st.markdown('이 3개 데이터의 경우, duration이 duration 항목 대신 rating 항목에 잘못 입력되어 있는 것으로 추정됨.')

    st.markdown('- rating별 개수 count')
    st.text(df['rating'].value_counts())

    st.markdown('- duration별 개수 count')
    st.text(df['duration'].value_counts())

    st.markdown('- rating 자리에 잘못 입력된 duration 데이터를 duration에 입력해주기')
    st.code("df.loc[df['duration'].isnull(), 'duration'] = df.loc[df['duration'].isnull(), 'rating']")
    df.loc[df['duration'].isnull(), 'duration'] = df.loc[df['duration'].isnull(), 'rating']

    st.markdown('- 잘못 입력된 rating 삭제')
    st.code("df.loc[df['director'] == 'Louis C.K.', 'rating'] = 'Unknown'")

    st.markdown('- rating의 결측치 제거')
    st.code('''
    df['rating'] = df['rating'].fillna('Unknown')
    df['rating'].isnull().value_counts()
    ''')
    df['rating'] = df['rating'].fillna('Unknown')

    st.markdown('- country의 결측치 제거')
    st.code('''
    df['country'] = df['country'].fillna('Unknown')
    df['country'].isnull().value_counts()
    ''')
    df['country'] = df['country'].fillna('Unknown')

elif i==2:
    st.subheader('(2) 칼럼별 데이터 시각화')
    st.markdown('- type / 영상 매체의 종류')

    fig, ax = plt.subplots()
    new_df.groupby('type').size().plot(kind='bar', ax = ax)
    ax.set_xlabel('type')
    ax.set_ylabel('number')
    st.pyplot(fig)

    st.markdown('- country / 제작 국가')
    st.markdown('1. 하나의 데이터에 여러 개의 국가가 할당된 경우가 있으므로 이를 분리해줍니다.')
    st.code('''
    country_list = new_df['country'].str.split(', ')
    country_df = country_list.explode()
    country_counts = country_df.value_counts()
    country_counts
    ''')
    country_list = new_df['country'].str.split(', ')
    st.text(country_list)
    country_df = country_list.explode()
    country_counts = country_df.value_counts()
    st.text(country_counts)

    st.markdown("2. 국가의 개수가 128개로 너무 많기 때문에, 2% 미만의 비중을 차지하는 국가의 경우 '기타(Other)' 항목으로 합쳐줍니다.")
    st.code('''
    total_count = country_counts.sum()
    threshold = total_count * 0.02
    # 임계값보다 작으면 기타 항목(Other)으로 분류
    other_countries = country_counts[country_counts < threshold]
    main_countries = country_counts[country_counts >= threshold]
    
    other_count = other_countries.sum()
    main_countries['Other'] = other_count
    ''')
    total_count = country_counts.sum()
    threshold = total_count * 0.02
    # 임계값보다 작으면 기타 항목(Other)으로 분류
    other_countries = country_counts[country_counts < threshold]
    main_countries = country_counts[country_counts >= threshold]

    other_count = other_countries.sum()
    main_countries['Other'] = other_count
    st.text(main_countries)

    st.markdown('3. 국가별 원그래프 그리기')
    labels = main_countries.index.tolist()
    explode = [0, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.20, 0.05]
    colors = sns.color_palette('hls', len(labels))

    fig, ax = plt.subplots()
    ax.pie(main_countries, labels=labels, explode=explode, colors=colors, autopct='%1.1f%%', startangle=130)
    ax.axis('equal')  # 원형을 유지
    st.pyplot(fig)

    st.markdown('  '
                '4. 기타 항목(Other)을 포함해 country를 다시 라벨링해줍니다.')
    st.code('''
        # 주어진 국가 목록과 'Other'로 라벨링할 기준 목록 정의
    countries = [
        'United States', 'India', 'None', 'United Kingdom', 'Canada',
        'France', 'Japan', 'Spain', 'South Korea', 'Germany'
    ]
    
    # country 열 수정 함수 정의
    def label_country(country):
        if country in countries:
            return country
        else:
            return 'Other'
    
    # country 열에 함수 적용
    new_df['country'] = new_df['country'].apply(label_country)
    
    # 수정된 데이터프레임 출력
    new_df
    ''')
    # 주어진 국가 목록과 'Other'로 라벨링할 기준 목록 정의
    countries = [
        'United States', 'India', 'None', 'United Kingdom', 'Canada',
        'France', 'Japan', 'Spain', 'South Korea', 'Germany'
    ]


    # country 열 수정 함수 정의
    def label_country(country):
        if country in countries:
            return country
        else:
            return 'Other'


    # country 열에 함수 적용
    new_df['country'] = new_df['country'].apply(label_country)

    # 수정된 데이터프레임 출력
    new_df

    st.markdown('- duration / 영상 매체 길이(시간(분) or 시즌 개수)')
    st.markdown('1. duration의 종류 확인')
    st.code('''
    pd.set_option('display.max_seq_items', None)
    df['duration'].value_counts().index.sort_values()
    ''')
    pd.set_option('display.max_seq_items', None)
    st.text(df['duration'].value_counts().index.sort_values())

    st.markdown('2. 시리즈물 / 시리즈물이 아닌 것의 비율 구한 뒤 시각화')
    st.code('''
    min_count = df['duration'].str.endswith('min').sum()
    season_count = df['duration'].str.endswith(('Season', 'Seasons')).sum()
    ''')
    st.code('''
    total_count = len(df)
    min_ratio = min_count / total_count
    season_ratio = season_count / total_count
    ''')
    st.code('''
    # 시리즈물 / 시리즈물이 아닌 것의 비율그래프
    labels = ['min', 'Season']
    ratios = [min_ratio, season_ratio]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, ratios)
    plt.xlabel('Duration Type')
    plt.ylabel('Ratio')
    plt.title('Ratio of "min" and "Season" in Duration')
    plt.ylim(0, 1)
    plt.show()
    ''')
    min_count = df['duration'].str.endswith('min').sum()
    season_count = df['duration'].str.endswith(('Season', 'Seasons')).sum()
    total_count = len(df)
    min_ratio = min_count / total_count
    season_ratio = season_count / total_count

    # 비율 그래프 생성
    labels = ['min', 'Season']
    ratios = [min_ratio, season_ratio]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, ratios)
    ax.set_xlabel('Duration Type')
    ax.set_ylabel('Ratio')
    ax.set_title('Ratio of "min" and "Season" in Duration')
    ax.set_ylim(0, 1)

    # Streamlit에 차트 표시
    st.pyplot(fig)

    st.markdown('3. 시리즈물의 시리즈 개수별 통계')
    st.code('''
    season_data = new_df[new_df['duration'].str.endswith(('Season', 'Seasons'))]
    ''')
    st.code('''
    season_counts = season_data['duration'].value_counts().reset_index()
    
    season_counts['category'] = season_counts['duration'].apply(
        lambda x: x if int(x.split()[0]) < 4 else '4 Seasons and above'
    )
    combined_counts = season_counts.groupby('category')['count'].sum().reset_index()
    combined_counts
    ''')
    st.code('''
    # 시리즈물의 시즌 개수별 막대그래프
    plt.figure(figsize=(10, 6))
    plt.bar(combined_counts['category'], combined_counts['count'], color='skyblue')
    plt.xlabel('Duration')
    plt.ylabel('Count')
    plt.title('Number of Seasons')
    plt.xticks(rotation=45)
    plt.show()
    ''')

    season_data = new_df[new_df['duration'].str.endswith(('Season', 'Seasons'))]
    season_counts = season_data['duration'].value_counts().reset_index()

    season_counts['category'] = season_counts['duration'].apply(
        lambda x: x if int(x.split()[0]) < 4 else '4 Seasons and above'
    )
    combined_counts = season_counts.groupby('category')['count'].sum().reset_index()
    combined_counts

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(combined_counts['category'], combined_counts['count'], color='skyblue')
    ax.set_xlabel('Duration')
    ax.set_ylabel('Count')
    ax.set_title('Number of Seasons')
    ax.set_xticks(combined_counts['category'])
    ax.set_xticklabels(combined_counts['category'], rotation=45)

    # Streamlit에 차트 표시
    st.pyplot(fig)

    st.markdown('4. 시리즈물이 아닌 것의 시간 길이별 통계')
    st.markdown('''
    ** short-term(60분 미만)  
    ** mid-term(60분 이상 120분 미만)  
    ** long-term(120분 이상)     
    ''')
    st.code('''
    def categorize_minutes(minutes):
      if minutes < 60:
        return 'short-term'
      elif 60 <= minutes < 120:
        return 'mid-term'
      else:
        return 'long-term'
    ''')
    st.code('''
    # 시리즈물이 아닌 것의 상영 길이별 통계

    min_data = df[df['duration'].str.endswith(('min'))].copy()
    min_data.loc[:, 'minutes'] = min_data['duration'].str.extract('(\d+)').astype(int)
    
    min_data.loc[:, 'category'] = min_data['minutes'].apply(categorize_minutes)
    
    category_counts = min_data['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    
    plt.figure(figsize=(10, 6))
    plt.bar(category_counts['category'], category_counts['count'], color='skyblue')
    plt.xlabel('Duration Category')
    plt.ylabel('Count')
    plt.title('Number of Entries per Duration Category')
    plt.xticks(rotation=45)
    plt.show()
    ''')

    def categorize_minutes(minutes):
        if minutes < 60:
            return 'short-term'
        elif 60 <= minutes < 120:
            return 'mid-term'
        else:
            return 'long-term'

    # 시리즈물이 아닌 것의 상영 길이별 통계
    min_data = new_df[new_df['duration'].str.endswith(('min'))].copy()
    min_data.loc[:, 'minutes'] = min_data['duration'].str.extract('(\d+)').astype(int)

    min_data.loc[:, 'category'] = min_data['minutes'].apply(categorize_minutes)

    category_counts = min_data['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    # 막대그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(category_counts['category'], category_counts['count'], color='skyblue')
    ax.set_xlabel('Duration Category')
    ax.set_ylabel('Count')
    ax.set_title('Number of Entries per Duration Category')
    ax.set_xticklabels(category_counts['category'], rotation=45)

    # Streamlit에 차트 표시
    st.pyplot(fig)

    st.markdown('5. df의 duration열을 다시 라벨링')
    st.code('''
    # duration 열 수정 함수 정의
    def label_duration(duration):
        if duration.endswith('min'):
            minutes = int(duration.split()[0])
            if minutes < 60:
                return 'short-term'
            elif 60 <= minutes < 120:
                return 'mid-term'
            else:
                return 'long-term'
        elif duration.endswith('Season') or duration.endswith('Seasons'):
            seasons = int(duration.split()[0])
            if seasons == 1:
                return '1 Season'
            elif seasons == 2:
                return '2 Seasons'
            elif seasons == 3:
                return '3 Seasons'
            else:
                return '4 Seasons and above'
        else:
            return duration  # 원본 데이터를 반환 (조건에 맞지 않는 경우)
    
    # duration 열에 함수 적용
    df['duration'] = df['duration'].apply(label_duration)
    
    # 수정된 데이터프레임 출력
    df
    ''')


    # duration 열 수정 함수 정의
    def label_duration(duration):
        if duration.endswith('min'):
            minutes = int(duration.split()[0])
            if minutes < 60:
                return 'short-term'
            elif 60 <= minutes < 120:
                return 'mid-term'
            else:
                return 'long-term'
        elif duration.endswith('Season') or duration.endswith('Seasons'):
            seasons = int(duration.split()[0])
            if seasons == 1:
                return '1 Season'
            elif seasons == 2:
                return '2 Seasons'
            elif seasons == 3:
                return '3 Seasons'
            else:
                return '4 Seasons and above'
        else:
            return duration  # 원본 데이터를 반환 (조건에 맞지 않는 경우)


    # duration 열에 함수 적용
    new_df['duration'] = new_df['duration'].apply(label_duration)

    # 수정된 데이터프레임 출력
    new_df

elif i == 3:
    st.subheader('(3) description의 코사인 유사도 기반 관련 콘텐츠 추천')
    st.markdown('- 본격적으로 type, country, duration을 반영한 추천시스템을 제작하기 전, description의 코사인 유사도만으로 콘텐츠를 추천하는 과정이다.')

    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 영화 제목을 인덱스로 매핑
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        # 영화 제목이 데이터프레임에 있는지 확인, 대소문자를 구분하지 않도록
        if title.lower() not in indices:
            return "The movie title is not in the dataset."

        title = title.lower()

        # 영화 인덱스를 가져오기
        idx = indices[title]

        # 모든 영화에 대한 유사도 점수 얻기
        sim_scores = list(enumerate(cosine_sim[idx]))

        # 유사도 점수에 따라 영화들을 정렬
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 가장 유사한 10개의 영화 선택
        sim_scores = sim_scores[1:11]

        # 유사한 영화들의 인덱스 얻기
        movie_indices = [i[0] for i in sim_scores]

        # 유사한 영화들의 제목 리턴
        return df['title'].iloc[movie_indices]

    # 알파벳 선택 및 영화 목록 업데이트
    selected_alphabet = st.selectbox('Select the first letter of the movie title:', list(alphabet_titles.keys()), key='selected_alphabet_1', on_change=update_movie_list_1)
    movie_title = st.selectbox('Select a movie title:', st.session_state['movie_title_list_1'])

    with st.form(key='movie_form'):
        submit_button = st.form_submit_button(label='Recommend')

    # 추천 결과 처리
    if submit_button:
        if movie_title:
            recommendations = get_recommendations(movie_title)
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.write('You may also like:')
                for i, title in enumerate(recommendations, 1):
                    st.write(f"{i}. {title}")
        else:
            st.write('Please select a movie title.')


elif i == 4:
    st.title('개인 맞춤 콘텐츠 추천 앱')

    # 데이터 초기화 버튼
    if st.button('Reset Data'):
        st.session_state['user_data'] = pd.DataFrame(
            columns=['title', 'rating', 'evaluation_date', 'description', 'country', 'duration', 'type'])
        st.success("Data has been reset.")

    # 알파벳 선택 및 영화 목록 업데이트
    selected_alphabet = st.selectbox('Select the first letter of the movie title:', list(alphabet_titles.keys()),
                                     key='selected_alphabet_2', on_change=update_movie_list_2)
    movie_title = st.selectbox('Select a movie title:', st.session_state['movie_title_list_2'])

    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 영화 제목을 인덱스로 매핑
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    def penalty(evaluation_date):
        current_date = datetime.now()
        days_passed = (current_date - evaluation_date).days
        return 1 / (1 + np.log1p(days_passed))

    def get_recommendations(title, cosine_sim=cosine_sim):
        if title not in indices:
            return []
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices].tolist()

    def get_top_similar_movies(movie_title, user_rating, eval_date, top_n=10):
        date_penalty = penalty(eval_date)
        similar_titles = get_recommendations(movie_title)
        recommendations = []
        for title in similar_titles:
            idx = indices[title]
            similarity_score = cosine_sim[indices[movie_title], idx]
            adjusted_score = user_rating * similarity_score * date_penalty
            recommendations.append((title, adjusted_score))
        return recommendations

    def create_preference_matrix(user_data):
        country_pref = user_data.groupby('country').apply(
            lambda x: (x['rating'] * x['evaluation_date'].apply(penalty)).sum()).to_dict()
        duration_pref = user_data.groupby('duration').apply(
            lambda x: (x['rating'] * x['evaluation_date'].apply(penalty)).sum()).to_dict()
        type_pref = user_data.groupby('type').apply(
            lambda x: (x['rating'] * x['evaluation_date'].apply(penalty)).sum()).to_dict()
        return country_pref, duration_pref, type_pref

    def calculate_inner_product(row, country_pref, duration_pref, type_pref):
        countries = str(row['country']).split(', ')
        country_scores = [country_pref.get(country, 0) for country in countries]
        duration_score = duration_pref.get(row['duration'], 0)
        type_score = type_pref.get(row['type'], 0)
        return sum(country_scores) + duration_score + type_score

    # 사용자가 본 영화 목록을 저장할 데이터프레임 초기화
    if 'user_data' not in st.session_state:
        st.session_state['user_data'] = pd.DataFrame(
            columns=['title', 'rating', 'evaluation_date', 'description', 'country', 'duration', 'type'])



    # 입력 폼 설정
    with st.form(key='movie_form'):
        movie_rating = st.select_slider('Rate the movie:', options=list(range(1, 11)), value=5)
        evaluation_date = st.date_input('Evaluation date:', datetime.now(), max_value=datetime.now().date())
        submit_button = st.form_submit_button(label='Add')

    # 입력 폼 제출 처리
    if submit_button:
        # df 데이터프레임에서 해당 영화 제목 가져오기
        movie_info = df[df['title'] == movie_title].iloc[0]
        new_data = pd.DataFrame({
            'title': [movie_info['title']],
            'rating': [movie_rating],
            'evaluation_date': [pd.to_datetime(evaluation_date)],
            'description': [movie_info['description']],
            'country': [movie_info['country']],
            'duration': [movie_info['duration']],
            'type': [movie_info['type']]
        })

        # 이미 평가된 영화 제목인 경우 기존 데이터 덮어쓰기
        st.session_state['user_data'] = st.session_state['user_data'][
            st.session_state['user_data']['title'] != movie_title]
        st.session_state['user_data'] = pd.concat([st.session_state['user_data'], new_data], ignore_index=True)

    # 현재 사용자 데이터 출력
    st.write('Your movie ratings:')
    st.write(st.session_state['user_data'])

    # 맞춤형 추천 계산
    if not st.session_state['user_data'].empty:
        recommendation_dict = {}
        for index, row in st.session_state['user_data'].iterrows():
            movie_title = row['title']
            user_rating = row['rating']
            eval_date = row['evaluation_date']
            recommendations = get_top_similar_movies(movie_title, user_rating, eval_date)
            for rec_movie_title, score in recommendations:
                if rec_movie_title in recommendation_dict:
                    recommendation_dict[rec_movie_title] += score
                else:
                    recommendation_dict[rec_movie_title] = score

        watched_movies = set(st.session_state['user_data']['title'])
        filtered_recommendations = {title: score for title, score in recommendation_dict.items() if
                                    title not in watched_movies}
        top_recommendations = sorted(filtered_recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
        recommended_movies = [rec[0] for rec in top_recommendations]

        # 사용자 선호도 행렬 생성
        country_pref, duration_pref, type_pref = create_preference_matrix(st.session_state['user_data'])
        top_10_data = df[df['title'].isin(recommended_movies)].copy()
        top_10_data['score'] = top_10_data.apply(
            lambda row: calculate_inner_product(row, country_pref, duration_pref, type_pref), axis=1)
        final_recommendations = top_10_data.sort_values(by='score', ascending=False).head(5)

        # Final top 5 recommendations 출력
        st.write('Final top 5 recommendations based on your preferences:')
        for i, row in enumerate(final_recommendations.itertuples(), 1):
            st.write(f"{i}. {row.title} (Score: {row.score:.2f})")