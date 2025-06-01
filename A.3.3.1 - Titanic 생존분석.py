# 필요한 라이브러리 임포트
dimport pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns # 시각화 개선을 위해 추가

# 0. 경고 메시지 무시 (선택 사항)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lifelines') # lifelines 관련 UserWarning 무시
warnings.filterwarnings('ignore', category=FutureWarning) # 가끔 pandas 등에서 발생하는 FutureWarning 무시


# 1. 데이터 로딩 및 전처리
try:
    # Kaggle에서 다운로드한 Titanic 훈련 데이터셋 파일 경로를 지정해야 합니다.
    # 예시: data = pd.read_csv('C:/Users/YourName/Downloads/titanic/train.csv')
    # 여기서는 'train.csv'가 현재 작업 디렉토리에 있다고 가정합니다.
    data = pd.read_csv('train.csv')
    print("데이터 로딩 성공.")
except FileNotFoundError:
    print("오류: 'train.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    print("데모를 위해 임의의 샘플 데이터를 생성합니다.")
    # 파일이 없을 경우를 대비한 작은 샘플 데이터 (실제 분석에는 부적합)
    data_dict = {
        'PassengerId': range(1, 21),
        'Survived': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'Pclass': [3, 1, 3, 1, 3, 1, 3, 2, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 2],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'Allen, Mr. William Henry', 'Moran, Mr. James', 'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard', 'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)', 'Nasser, Mrs. Nicholas (Adele Achem)', 'Sandstrom, Miss. Marguerite Rut', 'Bonnell, Miss. Elizabeth', 'Saundercock, Mr. William Henry', 'Andersson, Mr. Anders Johan', 'Vestrom, Miss. Hulda Amanda Adolfina', 'Hewlett, Mrs. (Mary D Kingcome) ', 'Rice, Master. Eugene', 'Williams, Mr. Charles Eugene', 'Vander Planke, Mrs. Julius (Emilie Vandemoortele)', 'Masselmani, Mrs. Fatima'],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female'],
        'Age': [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 2, np.nan, 31, np.nan],
        'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', '17463', '349909', '347742', '237736', 'PP 9549', '113783', 'A/5. 2151', '347082', '350406', '248706', '382652', '244373', '345763', '2649'],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708, 16.7, 26.55, 8.05, 31.275, 7.8542, 16, 29.125, 13, 18, 7.225],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan, np.nan, np.nan, 'G6', 'C103', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S', 'S', 'S', 'S', 'Q', 'S', 'S', 'C']
    }
    data = pd.DataFrame(data_dict)

# 생존분석에 필요한 변수 선택 및 결측치 처리
# .copy()를 사용하여 SettingWithCopyWarning 방지
df_survival = data[['Survived', 'Pclass', 'Sex', 'Age']].copy()

# 'Age' 결측치를 중앙값으로 대체
df_survival.loc[:, 'Age'] = df_survival['Age'].fillna(df_survival['Age'].median())

# 'Sex' 변수를 숫자형으로 변환 (male: 0, female: 1)
df_survival.loc[:, 'Sex'] = df_survival['Sex'].map({'male': 0, 'female': 1}).astype(int)

# 'Pclass'를 범주형으로 명시 (결과 해석 시 용이)
df_survival.loc[:, 'Pclass'] = df_survival['Pclass'].astype('category')

# 'Age'를 범주형으로 변환 (청소년, 성인, 노인)
# 나이 범주 정의 (예: 0-18: Youth, 19-60: Adult, 60+: Senior)
age_bins = [0, 18, 60, np.inf] # np.inf를 사용하여 60세 이상 모두 포함
age_labels = ['Youth', 'Adult', 'Senior']
df_survival.loc[:, 'AgeGroup'] = pd.cut(df_survival['Age'], bins=age_bins, labels=age_labels, right=False)

# 생존분석을 위한 시간(duration) 변수 생성
# 이 문제에서는 모든 승객의 관찰 시간이 동일(사고 시점까지)하다고 가정 -> Time = 1
df_survival.loc[:, 'Time'] = 1

print("\n데이터 전처리 후 샘플:")
print(df_survival.head())
print(f"\n결측치 확인:\n{df_survival.isnull().sum()}")

# 2. 카플란-마이어 생존 곡선 추정 및 시각화
kmf = KaplanMeierFitter()

# 그래프 스타일 설정
sns.set_style("whitegrid")

# Pclass 그룹별 생존 곡선
plt.figure(figsize=(8, 5))
for p_class in sorted(df_survival['Pclass'].cat.categories): # .cat.categories 사용
    group_data = df_survival[df_survival['Pclass'] == p_class]
    if not group_data.empty:
        kmf.fit(durations=group_data['Time'], event_observed=group_data['Survived'], label=f'Pclass {p_class}')
        kmf.plot_survival_function(ax=plt.gca()) # gca()로 현재 axes에 그림
plt.title('Kaplan-Meier Survival Curve by Pclass')
plt.xlabel('Time (Arbitrary Unit)')
plt.ylabel('Survival Probability')
plt.legend(title='Pclass')
plt.tight_layout()
plt.show()

# Sex 그룹별 생존 곡선
plt.figure(figsize=(8, 5))
for sex_val in sorted(df_survival['Sex'].unique()):
    group_data = df_survival[df_survival['Sex'] == sex_val]
    if not group_data.empty:
        label = 'Female' if sex_val == 1 else 'Male'
        kmf.fit(durations=group_data['Time'], event_observed=group_data['Survived'], label=label)
        kmf.plot_survival_function(ax=plt.gca())
plt.title('Kaplan-Meier Survival Curve by Sex')
plt.xlabel('Time (Arbitrary Unit)')
plt.ylabel('Survival Probability')
plt.legend(title='Sex')
plt.tight_layout()
plt.show()

# AgeGroup 그룹별 생존 곡선
plt.figure(figsize=(8, 5))
for age_group in age_labels: # 정의된 레이블 순서대로
    group_data = df_survival[df_survival['AgeGroup'] == age_group]
    if not group_data.empty:
        kmf.fit(durations=group_data['Time'], event_observed=group_data['Survived'], label=age_group)
        kmf.plot_survival_function(ax=plt.gca())
plt.title('Kaplan-Meier Survival Curve by Age Group')
plt.xlabel('Time (Arbitrary Unit)')
plt.ylabel('Survival Probability')
plt.legend(title='Age Group')
plt.tight_layout()
plt.show()


# 3. 로그-순위 검정
print("\n--- Log-rank Test Results ---")

# Pclass 그룹 간 비교 (모든 쌍 비교)
pclass_categories = sorted(df_survival['Pclass'].cat.categories)
if len(pclass_categories) > 1:
    for i in range(len(pclass_categories)):
        for j in range(i + 1, len(pclass_categories)):
            class1_label = pclass_categories[i]
            class2_label = pclass_categories[j]
            
            group1_data = df_survival[df_survival['Pclass'] == class1_label]
            group2_data = df_survival[df_survival['Pclass'] == class2_label]
            
            if not group1_data.empty and not group2_data.empty:
                results = logrank_test(group1_data['Time'], group2_data['Time'],
                                       event_observed_A=group1_data['Survived'], event_observed_B=group2_data['Survived'])
                print(f"Log-rank test (Pclass {class1_label} vs Pclass {class2_label}): Test Statistic={results.test_statistic:.2f}, p-value={results.p_value:.4f}")
            else:
                print(f"Log-rank test (Pclass {class1_label} vs Pclass {class2_label}): Not enough data for one or both groups.")
else:
    print("Pclass: Not enough groups to compare.")


# Sex 그룹 간 비교
sex_groups = sorted(df_survival['Sex'].unique())
if len(sex_groups) == 2: # 정확히 두 그룹일 때만 비교
    male_data = df_survival[df_survival['Sex'] == 0] # Male
    female_data = df_survival[df_survival['Sex'] == 1] # Female
    if not male_data.empty and not female_data.empty:
        results_sex = logrank_test(male_data['Time'], female_data['Time'],
                                   event_observed_A=male_data['Survived'], event_observed_B=female_data['Survived'])
        print(f"Log-rank test (Male vs Female): Test Statistic={results_sex.test_statistic:.2f}, p-value={results_sex.p_value:.4f}")
    else:
        print("Log-rank test (Male vs Female): Not enough data for one or both groups.")
else:
    print("Sex: Not enough groups to compare or more than two groups.")


# AgeGroup 그룹 간 비교 (모든 쌍 비교)
if len(age_labels) > 1:
    for i in range(len(age_labels)):
        for j in range(i + 1, len(age_labels)):
            age_group1_label = age_labels[i]
            age_group2_label = age_labels[j]

            group1_data = df_survival[df_survival['AgeGroup'] == age_group1_label]
            group2_data = df_survival[df_survival['AgeGroup'] == age_group2_label]
            
            if not group1_data.empty and not group2_data.empty:
                results_age = logrank_test(group1_data['Time'], group2_data['Time'],
                                           event_observed_A=group1_data['Survived'], event_observed_B=group2_data['Survived'])
                print(f"Log-rank test ({age_group1_label} vs {age_group2_label}): Test Statistic={results_age.test_statistic:.2f}, p-value={results_age.p_value:.4f}")
            else:
                print(f"Log-rank test ({age_group1_label} vs {age_group2_label}): Not enough data for one or both groups.")
else:
    print("AgeGroup: Not enough groups to compare.")

# 4. 결과 해석 (이 부분은 LLM이 텍스트로 생성할 내용입니다)
# print("\n--- Result Interpretation ---")
# print("카플란-마이어 생존 곡선과 로그-순위 검정 결과를 종합적으로 고려할 때...")
# print("- Pclass: ... (예: 1등석 승객의 생존율이 통계적으로 유의하게 높음)")
# print("- Sex: ... (예: 여성 승객의 생존율이 통계적으로 유의하게 높음)")
# print("- AgeGroup: ... (예: 청소년 그룹의 생존율이 통계적으로 유의하게 높음)")
