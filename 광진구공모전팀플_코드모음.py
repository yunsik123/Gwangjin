#%%라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame
from functools import reduce
#%%좌표를 이용해서 행정동구분

# shapefile 경로
# shapefile 불러오기
gdf = gpd.read_file('C:/Users/ncc05/Desktop/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp',encoding="cp949")

gdf_gwang = gdf[gdf['ADM_NM'].str.contains('중곡1동|중곡2동|중곡3동|중곡4동|능동|구의1동|구의2동|구의3동|광장동|자양1동|자양2동|자양3동|자양4동|화양동|군자동')]
#좌표계 변환
gdf_gwang.set_crs(epsg=5186,inplace=True,allow_override=True)
gdf_gwang.to_crs(epsg=4326,inplace=True)
gdf_gwang.crs


df_busstation=pd.read_excel("C:/Users/ncc05/Desktop/공모전데이터/서울시버스정류소202404.xlsx")
#광진구만 가져오기
df_busstation['ARS_ID'] = df_busstation['ARS_ID'].astype(str)
df_busstation = df_busstation[df_busstation['ARS_ID'].str.startswith('5')]


df_park=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/전국도시공원정보표준데이터.csv",encoding='cp949')
df_park.isna()
df_park = df_park[df_park['소재지지번주소'].notna() & df_park['소재지지번주소'].str.contains('서울특별시 광진구')]


df_bike=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/전국자전거도로표준데이터.csv",encoding='cp949')
df_bike=df_bike.query('시도명=="서울특별시"')


#%%데이터합치기
#1.버스
df_bus=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/동별개수/bus.csv",encoding="utf-8")
df_bus=df_bus.rename(columns={'ADM_NM': '동이름', '0': '버스'})
#2.공원
df_park=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/동별개수/park.csv",encoding="utf-8")
df_park=df_park.rename(columns={'ADM_NM': '동이름', '0': '공원'})

#.쓰레기통
df_bin=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/동별개수/가로쓰레기통_개수.csv",encoding="utf-8")
df_bin=df_bin.rename(columns={'Unnamed: 0': '동이름', '행정(동)': '가로쓰레기통'})




#광진구아파트 동이 없는 관계로 행정동 구역 없이 좌표로 구해보기
dff = pd.read_csv("C:/Users/ncc05/Desktop/apt_mst_info_202401.csv",encoding="cp949")
dff= dff[dff['rdnmadr'].str.contains('광진구')]
dff.info()#lo127 la 37

dff['geometry'] = dff.apply(lambda row: Point(row['lo'], row['la']), axis=1)
dff= GeoDataFrame(dff, geometry='geometry')
#광진구
gdf = gpd.read_file('C:/Users/ncc05/Desktop/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp',encoding="cp949")
gdf_gwang = gdf[gdf['ADM_NM'].str.contains('중곡1동|중곡2동|중곡3동|중곡4동|능동|구의1동|구의2동|구의3동|광장동|자양1동|자양2동|자양3동|자양4동|화양동|군자동')]
gdf_gwang.set_crs(epsg=5186,inplace=True,allow_override=True)
gdf_gwang.to_crs(epsg=4326,inplace=True)
#조인
dff = gpd.sjoin(gdf_gwang,dff)
dff=dff['ADM_NM'].value_counts().reset_index()
dff.columns = ['동이름', '아파트']

#의류수거함
df_cloth=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/동별개수/의류수거함_개수.csv",encoding="utf-8")
df_cloth.columns = ['동이름', '의류수거함']

#폐형광등건전지
df_battery=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/동별개수/폐형광등폐건전지수거함_개수.csv",encoding="utf-8")
df_battery.columns = ['동이름', '폐형광등건전지']

#인구

df_pop=pd.read_csv("C:/Users/ncc05/Desktop/공모전데이터/동별개수/주민등록인구_20240429232811.csv",encoding="utf-8")
df_pop = df_pop.iloc[:, [2, 3, 5]]
df_pop= df_pop.drop(df.index[:3])
df_pop.columns = ['동이름', '면적','인구수']
df_pop['인구수']=df_pop['인구수'].astype("float")
df_pop['인구밀도'] = df_pop['인구수'] / df_pop['면적']
df_pop['인구밀도']=df_pop['인구밀도'].round().astype(int)



#다합치기
data_frames = [df_bus, df_park, df_bin,dff,df_cloth,df_battery,df_pop]
merged_df = reduce(lambda left, right: pd.merge(left, right, on='동이름', how='outer'), data_frames)
merged_df.to_csv('merged.csv',index=False)

#%% pca와 군집분석을 이용한 행정동 선택(팀원)
#k-means
merged = pd.read_csv('merged.csv')
merged
ppl = ppl[['행정기관', '면적', '밀도']]
ppl = ppl.sort_values(by='행정기관')
ppl = ppl.reset_index(drop=True)
ppl
merged = merged.sort_values(by='동이름')
merged = merged.reset_index(drop=True)
merged
merged.to_csv('/content/drive/MyDrive/kwangjin/merged2.csv')
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = merged.drop(columns = ['동이름', '인구수', '면적'])

# 한글폰트 설치(mac)
!apt-get update -qq
!apt-get install fonts-nanum* -qq
# 한글폰트 깨짐방지(mac)
import matplotlib.font_manager as fm
import warnings
fe = fm.FontEntry(fname=r'/Users/hong-yujin/Library/Group Containers/UBF8T346G9.Office/FontCache/4/CloudFonts/NanumGothic/29131424179.ttf',
                  name='NanumGothic')
fm.fontManager.ttflist.insert(0,fe)
plt.rcParams.update({'font.size': 11, 'font.family': 'NanumGothic'})
# 한글폰트 설치(windows)
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
# 한글폰트 깨짐방지 (windows)
plt.rc('font', family='NanumBarunGothic')

corr = df.corr()

plt.figure(figsize=(8,6))

mask = np.zeros_like(corr, dtype = bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(data = corr,
            annot = True,
            mask = mask,
            fmt = '.2f',
            linewidths = 1.,
            cmap = 'PiYG')
plt.title('상관계수 히트맵')
plt.show()

from sklearn.cluster import KMeans

# k-means clustering
ks = range(1,11)
inertias = []
for k in ks:
  model = KMeans(n_clusters = k, n_init = 5)
  model.fit(df)
  inertias.append(model.inertia_)
  print('n_cluster: {}, inertia : {}'.format(k, model.inertia_))

# visualization
plt.figure(figsize=(15,6))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

from sklearn.metrics.cluster import silhouette_score

# 실루엣 분석을 사용하여 최적의 K값 탐색
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    silhouette_scores.append(score)

# 실루엣 분석 그래프 그리기
plt.plot(range(2, 11), silhouette_scores, marker='o')
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state = 0)

# 군집 어떻게 나눠져있는지 확인
kmeans.fit(df)
print(kmeans.labels_)
# 나눠진 군집 새로운 컬럼으로 추가
df['cluster'] = kmeans.labels_
df.head()
# 2차원 평면에서 그려주기 위해 2개의 차원으로 축소한 후, x좌표 y좌표로 표현
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(df)

df['pca_x'] = pca_transformed[:,0]
df['pca_y'] = pca_transformed[:,1]
df.head()

# 클러스터 값이 0,1,2,3인 경우마다 별도의 index로 추출
marker0_ind = df[df['cluster']==0].index
marker1_ind = df[df['cluster']==1].index
marker2_ind = df[df['cluster']==2].index
marker3_ind = df[df['cluster']==3].index

# 클러스터 값 0,1,2,3 해당 index로 차원축소한 x좌표 y좌표 값 추출. o,s,^로 마커 표시
plt.scatter(x=df.loc[marker0_ind, 'pca_x'], y=df.loc[marker0_ind, 'pca_y'], marker='o')
plt.scatter(x=df.loc[marker1_ind, 'pca_x'], y=df.loc[marker1_ind, 'pca_y'], marker='s')
plt.scatter(x=df.loc[marker2_ind, 'pca_x'], y=df.loc[marker2_ind, 'pca_y'], marker='^')
plt.scatter(x=df.loc[marker3_ind, 'pca_x'], y=df.loc[marker3_ind, 'pca_y'], marker='*')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('4 Clusters Visualization by 2 PCA Components')
df['행정동'] = merged['동이름']
df[df['cluster']==2]


#kmedoids
df = merged.drop(columns = ['동이름', '인구수', '면적'])
df
from sklearn.preprocessing import StandardScaler

scaler =  StandardScaler()
df_scale = scaler.fit_transform(df)
df_scaler = pd.DataFrame(df_scale)

!pip install scikit-learn-extra

from sklearn_extra.cluster import KMedoids
k = 3  # 클러스터의 개수
kmedoids = KMedoids(n_clusters=k, random_state=42)
labels = kmedoids.fit_predict(df_scaler)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(df_scaler)

df_scaler['pca_x'] = pca_transformed[:,0]
df_scaler['pca_y'] = pca_transformed[:,1]
df_scaler.head()


# 클러스터 값이 0,1,2인 경우마다 별도의 index로 추출
marker0_ind = df_scaler[df_scaler['Cluster']==0].index
marker1_ind = df_scaler[df_scaler['Cluster']==1].index
marker2_ind = df_scaler[df_scaler['Cluster']==2].index
#marker3_ind = df_scaler[df_scaler['Cluster']==3].index

# 클러스터 값 0,1,2 해당 index로 차원축소한 x좌표 y좌표 값 추출. o,s,^로 마커 표시
plt.scatter(x=df_scaler.loc[marker0_ind, 'pca_x'], y=df_scaler.loc[marker0_ind, 'pca_y'], marker='o')
plt.scatter(x=df_scaler.loc[marker1_ind, 'pca_x'], y=df_scaler.loc[marker1_ind, 'pca_y'], marker='s')
plt.scatter(x=df_scaler.loc[marker2_ind, 'pca_x'], y=df_scaler.loc[marker2_ind, 'pca_y'], marker='^')
#plt.scatter(x=df_scaler.loc[marker3_ind, 'pca_x'], y=df_scaler.loc[marker3_ind, 'pca_y'], marker='*')


plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('4 Clusters Visualization by 2 PCA Components')

df_scaler['행정동'] = merged['동이름']
df_scaler[df_scaler['Cluster']==2]

#gmm
merged = pd.read_csv('merged.csv')
merged

ppl = ppl[['행정기관', '면적', '밀도']]
ppl = ppl.sort_values(by='행정기관')
ppl = ppl.reset_index(drop=True)
ppl

merged = merged.sort_values(by='동이름')
merged = merged.reset_index(drop=True)
merged

# 전처리 데이터 동이름 기준으로 정렬, 면적/인구밀도 변수 추가
merged[['면적', '인구밀도']] = ppl[['면적', '밀도']]
merged.head()

df = merged.drop(columns = ['동이름', '인구수', '면적'])
df


# 스케일링

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scale = scaler.fit_transform(df)
df = pd.DataFrame(df_scale)
df

from sklearn.mixture import GaussianMixture

n_components = np.arange(1, 11)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df)
          for n in n_components]

plt.plot(n_components, [m.bic(df) for m in models], label='BIC')
plt.plot(n_components, [m.aic(df) for m in models], label='AIC')

plt.legend(loc='best')
plt.xlabel('n_components');

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, n_init=10, random_state=1)
gmm.fit(df)

print(gmm.bic(df))
print(gmm.aic(df))

gmm_label = gmm.fit_predict(df)

df['cluster'] = gmm_label
df

# 2차원 평면에서 그려주기 위해 2개의 차원으로 축소한 후, x좌표 y좌표로 표현
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(df)

df['pca_x'] = pca_transformed[:,0]
df['pca_y'] = pca_transformed[:,1]
df.head()

# 클러스터 값이 0,1,2인 경우마다 별도의 index로 추출
marker0_ind = df[df['cluster']==0].index
marker1_ind = df[df['cluster']==1].index
marker2_ind = df[df['cluster']==2].index

# 클러스터 값 0,1,2 해당 index로 차원축소한 x좌표 y좌표 값 추출. o,s,^로 마커 표시
plt.scatter(x=df.loc[marker0_ind, 'pca_x'], y=df.loc[marker0_ind, 'pca_y'], marker='o')
plt.scatter(x=df.loc[marker1_ind, 'pca_x'], y=df.loc[marker1_ind, 'pca_y'], marker='s')
plt.scatter(x=df.loc[marker2_ind, 'pca_x'], y=df.loc[marker2_ind, 'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusters Visualization by 2 PCA Components')


df['행정동'] = merged['동이름']
df[df['cluster']==1]


#Hierachical
merged = pd.read_csv('merged.csv')
df = merged.drop(columns = ['동이름', '인구수', '면적'])
df

from sklearn.preprocessing import StandardScaler

scaler =  StandardScaler()
df_scale = scaler.fit_transform(df)
df_scaler = pd.DataFrame(df_scale)

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(df_scaler, 'single')

labelList = list(range(1,11))

plt.figure(figsize=(7,5))
dendrogram(linked,
          orientation='top',
          distance_sort='descending',
          show_leaf_counts=True)

plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3,
                                  affinity='euclidean', linkage='ward')
a = cluster.fit_predict(df_scaler)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(df_scaler)

df_scaler['pca_x'] = pca_transformed[:,0]
df_scaler['pca_y'] = pca_transformed[:,1]
df_scaler.head()

df_scaler['cluster'] = a
df_scaler.head()

# 클러스터 값이 0,1,2인 경우마다 별도의 index로 추출
marker0_ind = df_scaler[df_scaler['cluster']==0].index
marker1_ind = df_scaler[df_scaler['cluster']==1].index
marker2_ind = df_scaler[df_scaler['cluster']==2].index

# 클러스터 값 0,1,2 해당 index로 차원축소한 x좌표 y좌표 값 추출. o,s,^로 마커 표시
plt.scatter(x=df_scaler.loc[marker0_ind, 'pca_x'], y=df_scaler.loc[marker0_ind, 'pca_y'], marker='o')
plt.scatter(x=df_scaler.loc[marker1_ind, 'pca_x'], y=df_scaler.loc[marker1_ind, 'pca_y'], marker='s')
plt.scatter(x=df_scaler.loc[marker2_ind, 'pca_x'], y=df_scaler.loc[marker2_ind, 'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusters Visualization by 2 PCA Components')

df_scaler['행정동'] = merged['동이름']
df_scaler[df_scaler['cluster']==0]
#%%건물정보데이터중 광진구만 가져옴
import geopandas as gpd
gdf = gpd.read_file('C:/Users/ncc05/Desktop/공모전데이터/동별개수/AL_D010_11_20240404/AL_D010_11_20240404.shp',encoding='cp949')

#for index, row in gdf.iterrows():
#    # 위치 좌표 출력 (여기에서는 중심 좌표를 출력하도록 하였습니다)
#    print(f'Feature {index + 1}의 중심 좌표: {row.geometry.centroid}')



#A1이 중요하고 이게 위치정보를 담고있음 A24는 건물명임 A4는 법정동임
A=gdf[['A1','A4']]
#A4이용해서 광진구만 뽑아줄수있음
A= A[gdf['A4'].str.contains('광진구')]
#소수점으로 짤라야하나..
#x좌표 y좌표 뽑아줌 #EPSG:5186임
A['x좌표'] = A['A1'].apply(lambda x: int(str(x)[4:12])).tolist()
A['y좌표'] = A['A1'].apply(lambda x: int(str(x)[12:20])).tolist()

#소수점변환
def convert_to_float_with_decimal(num):
    return float(num) / 100
A['x좌표'] = A['x좌표'].apply(convert_to_float_with_decimal)
A['y좌표'] = A['y좌표'].apply(convert_to_float_with_decimal)

#20703202
#45047183
A.to_csv("A.csv",index=False)

#la 위도 y좌표 37
#여기부터실행
A['geometry'] = A.apply(lambda row: Point(row['x좌표'], row['y좌표']), axis=1)
A = gpd.GeoDataFrame(A, geometry='geometry')
#변환
A.set_crs(epsg=5174,inplace=True,allow_override=True)
A.to_crs(epsg=4326,inplace=True)
A['x좌표2'] = A['geometry'].apply(lambda geom: geom.x)
A['y좌표2'] = A['geometry'].apply(lambda geom: geom.y)




#광진구
gdff = gpd.read_file('C:/Users/ncc05/Desktop/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp',encoding="cp949")
gdf_gwang = gdff[gdff['ADM_NM'].str.contains('중곡1동')]#중곡동만
gdf_gwang.set_crs(epsg=5186,inplace=True,allow_override=True)
gdf_gwang.to_crs(epsg=4326,inplace=True)
#조인
df = gpd.sjoin(gdf_gwang,A)
df= gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x좌표2'], df['y좌표2']))


#%%유전알고리즘과 mclp
#최소거리까지고려
#해의 생성 과정에서 최소 거리를 보장하는 방법
#해가 생성된 후에도 최소 거리를 유지하도록 하는 방법
#자양3동
df=df1[['long','lat']]
df.reset_index(inplace=True)
df1=df
df1.reset_index(inplace=True)

import random
import numpy as np

# 두 지점 간의 거리를 계산하는 함수
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 초기 해 집합을 생성하는 함수 (최소 거리를 충족하도록)
def generate_initial_solution(N, M, min_distance, points):
    solution = []
    while len(solution) < M:
        candidate = random.randint(0, N - 1)
        if all(calculate_distance(points[candidate], points[s]) >= min_distance for s in solution):
            solution.append(candidate)
    return solution

# 개체의 적합도(커버하는 위치의 수)를 계산하는 함수
def compute_fitness(solution, points, S):
    covered = set()
    for idx in solution:
        for jdx, point in enumerate(points):
            if jdx not in covered and calculate_distance(points[idx], point) <= S:
                covered.add(jdx)
    return len(covered)

# 유전 알고리즘을 사용하여 MCLP 문제를 해결하는 함수
def genetic_algorithm(points, S, population_size, generations, M, min_distance):
    N = len(points)
    population = [generate_initial_solution(N, M, min_distance, points) for _ in range(population_size)]

    for _ in range(generations):
        population = sorted(population, key=lambda x: compute_fitness(x, points, S), reverse=True)
        selected_parents = population[:population_size // 2]

        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)
            crossover_point = random.randint(0, M - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            if random.random() < 0.1:  # 돌연변이 확률
                mutation_point = random.randint(0, M - 1)
                child[mutation_point] = random.randint(0, N - 1)
            new_population.append(child)

        population = new_population

    best_solution = max(population, key=lambda x: compute_fitness(x, points, S))
    best_fitness = compute_fitness(best_solution, points, S)
    return best_solution, best_fitness

# 데이터프레임 정의
# 예시로 데이터프레임의 열 이름이 'lat'와 'long'이라고 가정
# 실제 데이터프레임의 열 이름에 맞게 수정하세요
df = df[['lat', 'long']]

# 최대 커버리지 반경 설정 (미터)
S = 0.005
# 초기 해 집합의 크기 설정
M = 2
# 유전 알고리즘 매개 변수 설정
population_size = 100
generations = 100
# 최소 거리 설정 (예: 100미터)
min_distance = 0.005

# 유전 알고리즘으로 MCLP 문제 해결
best_solution, best_fitness = genetic_algorithm(df.values, S, population_size, generations, M, min_distance)

# 결과 출력
print("최적의 해:", best_solution)
print("해의 적합도:", best_fitness)

#최적의 해: [444, 362]#해의 적합도: 965

#구의3동
df=df2[['long','lat']]
df.reset_index(inplace=True)
df2=df
df2.reset_index(inplace=True)
import random
import numpy as np

# 두 지점 간의 거리를 계산하는 함수
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 초기 해 집합을 생성하는 함수 (최소 거리를 충족하도록)
def generate_initial_solution(N, M, min_distance, points):
    solution = []
    while len(solution) < M:
        candidate = random.randint(0, N - 1)
        if all(calculate_distance(points[candidate], points[s]) >= min_distance for s in solution):
            solution.append(candidate)
    return solution

# 개체의 적합도(커버하는 위치의 수)를 계산하는 함수
def compute_fitness(solution, points, S):
    covered = set()
    for idx in solution:
        for jdx, point in enumerate(points):
            if jdx not in covered and calculate_distance(points[idx], point) <= S:
                covered.add(jdx)
    return len(covered)

# 유전 알고리즘을 사용하여 MCLP 문제를 해결하는 함수
def genetic_algorithm(points, S, population_size, generations, M, min_distance):
    N = len(points)
    population = [generate_initial_solution(N, M, min_distance, points) for _ in range(population_size)]

    for _ in range(generations):
        population = sorted(population, key=lambda x: compute_fitness(x, points, S), reverse=True)
        selected_parents = population[:population_size // 2]

        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)
            crossover_point = random.randint(0, M - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            if random.random() < 0.1:  # 돌연변이 확률
                mutation_point = random.randint(0, M - 1)
                child[mutation_point] = random.randint(0, N - 1)
            new_population.append(child)

        population = new_population

    best_solution = max(population, key=lambda x: compute_fitness(x, points, S))
    best_fitness = compute_fitness(best_solution, points, S)
    return best_solution, best_fitness

# 데이터프레임 정의
# 예시로 데이터프레임의 열 이름이 'lat'와 'long'이라고 가정
# 실제 데이터프레임의 열 이름에 맞게 수정하세요
df = df[['lat', 'long']]

# 최대 커버리지 반경 설정 (미터)
S = 0.005
# 초기 해 집합의 크기 설정
M = 2
# 유전 알고리즘 매개 변수 설정
population_size = 100
generations = 100
# 최소 거리 설정 (예: 100미터)
min_distance = 0.005

# 유전 알고리즘으로 MCLP 문제 해결
best_solution, best_fitness = genetic_algorithm(df.values, S, population_size, generations, M, min_distance)

# 결과 출력
print("최적의 해:", best_solution)
print("해의 적합도:", best_fitness)

#최적의 해: [556, 249]#해의 적합도: 1090

#광장동
df=df3[['long','lat']]
df.reset_index(inplace=True)
df3=df
df3.reset_index(inplace=True)
import random
import numpy as np

# 두 지점 간의 거리를 계산하는 함수
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 초기 해 집합을 생성하는 함수 (최소 거리를 충족하도록)
def generate_initial_solution(N, M, min_distance, points):
    solution = []
    while len(solution) < M:
        candidate = random.randint(0, N - 1)
        if all(calculate_distance(points[candidate], points[s]) >= min_distance for s in solution):
            solution.append(candidate)
    return solution

# 개체의 적합도(커버하는 위치의 수)를 계산하는 함수
def compute_fitness(solution, points, S):
    covered = set()
    for idx in solution:
        for jdx, point in enumerate(points):
            if jdx not in covered and calculate_distance(points[idx], point) <= S:
                covered.add(jdx)
    return len(covered)

# 유전 알고리즘을 사용하여 MCLP 문제를 해결하는 함수
def genetic_algorithm(points, S, population_size, generations, M, min_distance):
    N = len(points)
    population = [generate_initial_solution(N, M, min_distance, points) for _ in range(population_size)]

    for _ in range(generations):
        population = sorted(population, key=lambda x: compute_fitness(x, points, S), reverse=True)
        selected_parents = population[:population_size // 2]

        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)
            crossover_point = random.randint(0, M - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            if random.random() < 0.1:  # 돌연변이 확률
                mutation_point = random.randint(0, M - 1)
                child[mutation_point] = random.randint(0, N - 1)
            new_population.append(child)

        population = new_population

    best_solution = max(population, key=lambda x: compute_fitness(x, points, S))
    best_fitness = compute_fitness(best_solution, points, S)
    return best_solution, best_fitness

# 데이터프레임 정의
# 예시로 데이터프레임의 열 이름이 'lat'와 'long'이라고 가정
# 실제 데이터프레임의 열 이름에 맞게 수정하세요
df = df[['lat', 'long']]

# 최대 커버리지 반경 설정 (미터)
S = 0.005
# 초기 해 집합의 크기 설정
M = 2
# 유전 알고리즘 매개 변수 설정
population_size = 100
generations = 100
# 최소 거리 설정 (예: 100미터)
min_distance = 0.005

# 유전 알고리즘으로 MCLP 문제 해결
best_solution, best_fitness = genetic_algorithm(df.values, S, population_size, generations, M, min_distance)

# 결과 출력
print("최적의 해:", best_solution)
print("해의 적합도:", best_fitness)

# [156, 797] 1000













#%%그리디 알고리즘과 p-median방법(팀원)

# 기존 재활용수거기 데이터
original = pd.read_csv('original.csv')
original

recycle = original[['name', 'long', 'lat']]
recycle


import pandas as pd
#import pulp
from haversine import haversine, Unit

import folium
from folium.features import CustomIcon
map = folium.Map(location = [37.53573889, 127.0845333], zoom_start = 11)
map

for _, row in original.iterrows():
  icon_image = 'icon.png'
  icon = CustomIcon(icon_image, icon_size=(25,25))
  popup = folium.Popup(row['name'], max_width=200)
  folium.Marker(location=[row['lat'], row['long']], popup=popup, icon=icon).add_to(map)

from shapely.geometry import Point, Polygon, LineString

import geopandas as gpd
A=pd.read_csv("kwangjin_code.csv")
A['geometry'] = A.apply(lambda row: Point(row['x좌표'], row['y좌표']), axis=1)
A = gpd.GeoDataFrame(A, geometry='geometry')
#변환
A.set_crs(epsg=5174,inplace=True,allow_override=True)
A.to_crs(epsg=4326,inplace=True)
A['x좌표2'] = A['geometry'].apply(lambda geom: geom.x)
A['y좌표2'] = A['geometry'].apply(lambda geom: geom.y)

gdff = gpd.read_file('/content/drive/MyDrive/kwangjin/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp',encoding="cp949")
gdf_gwang2 = gdff[gdff['ADM_NM'].str.contains('광장동')]
gdf_gwang2.set_crs(epsg=5186,inplace=True,allow_override=True)
gdf_gwang2.to_crs(epsg=4326,inplace=True)
#조인
df3 = gpd.sjoin(gdf_gwang2,A)
df3 = gpd.GeoDataFrame(df3, geometry=gpd.points_from_xy(df3['x좌표2'], df3['y좌표2']))


target = pd.concat([df1, df2, df3])
target.to_csv('/content/drive/MyDrive/kwangjin/target.csv')

target = pd.read_csv('target.csv')
target

target1 = target[['x좌표2', 'y좌표2']]
target1


target1.reset_index(inplace=True)
target1 = target1[['x좌표2', 'y좌표2']]

target1.columns = ['long', 'lat']
target1

### 10개의 기존 재활용품 회수기와 3289개의 수요지 간 수직 거리 distance에 저장
recycle = df[['long', 'lat']]
recycle

recycle = recycle[['long', 'lat']]


target = pd.read_csv('target.csv')
target1 = target[['x좌표2', 'y좌표2']]
target1.columns = ['long', 'lat']
target1


distances = []
original_nodes = []
destination_nodes = []

for index1, row1 in recycle.iterrows():
    for index2, row2 in target1.iterrows():
        start = (row1['lat'], row1['long'])
        end = (row2['lat'], row2['long'])
        distance = haversine(start, end, unit=Unit.METERS)
        distances.append(distance)
        original_nodes.append(index1)
        destination_nodes.append(index2)

# 새로운 데이터프레임 생성
new_df = pd.DataFrame({
    'original': original_nodes,  # original 노드
    'destination': destination_nodes,  # 목적지 노드
    'distance': distances  # 수직거리
})

print(new_df)


distances = []
original_nodes = []
destination_nodes = []

for index1, row1 in recycle.iterrows():
    for index2, row2 in target1.iterrows():
        start = (row1['lat'], row1['long'])
        end = (row2['lat'], row2['long'])
        distance = haversine(start, end, unit=Unit.METERS)
        distances.append(distance)
        original_nodes.append(index1)
        destination_nodes.append(index2)

# 새로운 데이터프레임 생성
new_df = pd.DataFrame({
    'original': original_nodes,  # original 노드
    'destination': destination_nodes,  # 목적지 노드
    'distance': distances  # 수직거리
})

print(new_df)
new_df
original['최소거리']=[0.0]*len(original.index)
original.최소거리.describe()

# 최소거리 평균 643m

import random

def solve_p_median_greedy_with_distance_constraint(df, p, min_distance):
    # 각 노드들의 번호를 추출하여 노드 리스트 생성
    locations = set(df['original']).union(set(df['destination']))
    selected_locations = list(locations)

    # Greedy 알고리즘에 따라 위치 선택
    medians = []
    while len(medians) < p:
        remaining_locations = [loc for loc in selected_locations if loc not in medians]
        best_location = None
        best_cost = float('inf')
        for location in remaining_locations:
            # 현재 위치를 추가했을 때의 비용 계산
            current_cost = sum(df.loc[(df['original'] == location) | (df['destination'] == location), 'distance'])
            # 최소 거리 제약을 추가하여 선택
            if all(df.loc[(df['original'] == med) | (df['destination'] == med), 'distance'].min() >= min_distance for med in medians) and current_cost < best_cost:
                best_location = location
                best_cost = current_cost
        if best_location:
            medians.append(best_location)
        else:
            # 만약 적절한 위치가 없으면 임의의 위치 선택
            medians.append(random.choice(remaining_locations))

    return medians

# Greedy 알고리즘으로 p-median 문제 풀기
optimal_locations_greedy = solve_p_median_greedy_with_distance_constraint(new_df, 3, 643)
print(optimal_locations_greedy)

print(target1.loc[2049])
print(target1.loc[261])
print(target1.loc[2659])

new_trash = pd.DataFrame({'name':['구의로 58', '자양번영로3길 12-8', '아차산로 599'],
                             'long':['127.090418', '127.074707', '127.104442'],
                             'lat':['37.543116', '37.532080', '37.546478']})

new_trash

import requests

KAKAO_REST_API_KEY = 'bc7e478d539807f1bb6a1c903964e224'

def convert_coordinates_to_address(lat, lng):

    y, x = str(lat), str(lng)
    url = 'https://dapi.kakao.com/v2/local/geo/coord2address.json?x={}&y={}'.format(x, y)
    header = {'Authorization': 'KakaoAK ' + KAKAO_REST_API_KEY}

    r = requests.get(url, headers=header)

    if r.status_code == 200:
        road_address = r.json()["documents"][0]["road_address"]['address_name']
        bunji_address = r.json()["documents"][0]["address"]['address_name']
    else:
        return None

    return road_address, bunji_address

convert_coordinates_to_address(37.546478, 127.104442)

map = folium.Map(location = [37.53573889, 127.0845333], zoom_start = 11)
map

for _, row in new_trash.iterrows():
  icon_image = 'icon.png'
  icon = CustomIcon(icon_image, icon_size=(45,45))
  popup = folium.Popup(row['name'], max_width=200)
  folium.Marker(location=[row['lat'], row['long']], popup=popup, icon=icon).add_to(map)

map







#%%좌표시각화
df = pd.concat([df3.loc[[156]],df3.loc[[797]], df2.loc[[556]],df2.loc[[249]],df1.loc[[444]],df1.loc[[362]]], ignore_index=True)


#0.005도는 약 555m
map_gwangjin = folium.Map(location=[37.5425, 127.0831], zoom_start=13)

#점찍기
for index, row in df.iterrows():
    folium.Marker([row['lat'], row['long']],icon=folium.Icon(color='blue')).add_to(map_gwangjin)

# 원그리기
for index, row in df.iterrows():
    folium.Circle([row['lat'], row['long']],radius=50,color='blue',fill_color='blue').add_to(map_gwangjin)
map_gwangjin.save('map_with_markers.html')








