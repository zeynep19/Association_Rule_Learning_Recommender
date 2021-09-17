# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
# Çıktının tek bir satırda olmasını sağlar.
from mlxtend.frequent_patterns import apriori, association_rules

###################################################################
# Görev-1 Veri Ön İşleme İşlemlerini Gerçekleştiriniz
###################################################################
df_ = pd.read_excel("pythonProject/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

###################################################################
# Görev-2 Germany müşterileri üzerinden birliktelik kuralları üretiniz.
###################################################################
# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)

df_gr = df[df['Country'] == "Germany"]

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
        # unstack ile satırlara ürünlerin faturalarını aldık, sütunlara da ürünleri aldık.
        # iloc ile index seçimi yaptık.

# Boolean biçimli tablo, tüm satılan ürünlerin özellik olarak (sütunlarda) yazıldığı ve her özelliğin var olma/olmama
# durumlarının gösterildiği tablodur. Bu tablo aşağıdaki gibidir:
gr_inv_pro_df = create_invoice_product_df(df_gr)
gr_inv_pro_df.head()

# Birliktelik Kurallarının Çıkarılması

# Frekans(tekrar sayısı): Bir ürünün toplam satış miktarı

# Support: X ürünü ile Y ürününün birlikte görülme olasığıdır. Sınırları 0–1 arasındadır.
# Support = Frekans / Toplam Girdi
# Minimum Sıpport: Minimum Support değeri ise bizim tarafımızdan belirleniyor olup değişiklik gösterebilir.

# Apriori algoritmasının ilk adımı, elimizde bulunan verilerdeki her ürünün frekans değerinin (tekrar sayısının) bulunup,
# support değerlerinin hesaplanmasıdır. Ardından Minimum Support değerine eşit veya üstünde bir Support değerine sahip
# ürünlerimiz ile yeni bir tablo oluşturacağız sonra tekrar kombinasyonlarını bulacağız ve tekrar tabloları oluşturacağız.
# Bu işlem, oluşturabildiğimiz en yüksek değerli tablo olana kadar devam edecektir. Oluşturduğumuz son
# Birliktelikler Tablosu’ndan kurallar çıkarımı bu adımda gerçekleşecektir. Birliktelik Kuralları değerlerinden

# Confidence: X satıldığında Y nin satılması olasılığı. Sınırları 0–1 arasındadır.
# Lift değeri: X satıldığında Y nin satılması olasılığı liftkadar artar. 0 ile ∞(sonsuz) arasındadır.

frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head(500)
###################################################################
# Görev-3 ID'leri verilen ürünlerin isimleri nelerdir?
###################################################################
# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

product_id = 21987
check_id(df_gr, product_id)
# ['PACK OF 6 SKULL PAPER CUPS']

product_id = 23235
check_id(df_gr, product_id)
# ['STORAGE TIN VINTAGE LEAF']

product_id = 22747
check_id(df_gr, product_id)
# ["POPPY'S PLAYHOUSE BATHROOM"]

###################################################################
# Görev-4 Sepetteki kullanıcılara için ürün önerisi yapınız.
# Görev-5 Önerilen ürünlerin isimleri nelerdir?
###################################################################
def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

# product_id = 22747 ["POPPY'S PLAYHOUSE BATHROOM"] ürününü sepete ekleyen için 1 ürün önerisi
arl_recommender(rules, 22747, 1)
# [22659]
check_id(df_gr, 22659)
# ['LUNCH BOX I LOVE LONDON']

# product_id = 23235 ['STORAGE TIN VINTAGE LEAF'] ürününü sepete ekleyen için 2 ürün önerisi
arl_recommender(rules, 23235, 2)
# [22726, 23206]
check_id(df_gr, 22726)
# ['ALARM CLOCK BAKELIKE GREEN']
check_id(df_gr, 23206)
# ['LUNCH BAG APPLE DESIGN']