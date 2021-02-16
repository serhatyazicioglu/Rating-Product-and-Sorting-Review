import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Görev-1: Bir ürünün rating’ini, güncel yorumlara göre hesaplayınız ve eski rating ile kıyaslayınız.

# Adım-1: df_sub.csv veri setini okutunuz.

df_sub = pd.read_csv("datasets/df_sub.csv")
df_sub.head()

# Adım-2: Ürünün ortalama puanı nedir?

df_sub["overall"].mean()

# Adım-3: Tarihe göre ağırlıklı puan ortalaması hesaplayınız.

df_sub['reviewTime'] = pd.to_datetime(df_sub['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df_sub["day_dif"] = (current_date - df_sub['reviewTime']).dt.days

a = df_sub["day_dif"].quantile(0.25)
b = df_sub["day_dif"].quantile(0.50)
c = df_sub["day_dif"].quantile(0.75)

# Adım-4: a,b,c değerlerine göre ağırlıklı puanı hesaplayınız.

df_sub["date_rating"] = df_sub.loc[df_sub["day_dif"] <= a, "overall"].mean() * 28 / 100 + \
                        df_sub.loc[(df_sub["day_dif"] > a) & (df_sub["day_dif"] <= b), "overall"].mean() * 26 / 100 + \
                        df_sub.loc[(df_sub["day_dif"] > b) & (df_sub["day_dif"] <= c), "overall"].mean() * 24 / 100 + \
                        df_sub.loc[(df_sub["day_dif"] > c), "overall"].mean() * 22 / 100

df_sub.sort_values("date_rating", ascending=False).head()

# Görev-2: Product tanıtım sayfasında görüntülenecek ilk 20 yorumu belirleyiniz.

# Adım-1: Helpful değişkeni içerisinden 3 değişken türetiniz.
df_sub["helpful"].head()

df_sub["helpful"] = df_sub["helpful"].apply(lambda x: x[1:-1].split(","))
df_sub["helpful_yes"] = [int(i[0]) for i in df_sub["helpful"]]
df_sub["helpful_no"] = [(int(i[1]) - int(i[0])) for i in df_sub["helpful"]]
df_sub["total_vote"] = [int(i[1]) for i in df_sub["helpful"]]

df = df_sub[["reviewerName", "overall", "helpful_yes", "helpful_no", "total_vote", "day_dif"]]


# Adım-2: score_pos_neg_diff'a göre skorlar oluşturunuz.
# Ardından; df_sub içerisinde score_pos_neg_diff ismiyle kaydediniz.

def score_pos_neg_diff(pos, neg):
    return pos - neg


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)


# Adım-3: score_average_rating'a göre skorlar oluşturunuz
# Ardından; df_sub içerisinde score_average_rating ismiyle kaydediniz.

def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


# Adım-4: wilson_lower_bound'a göre skorlar oluşturunuz.
# Ardından; df_sub içerisinde wilson_lower_bound ismiyle kaydediniz.

def wilson_lower_bound(pos, neg, confidence=0.95):
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Adım-5: Ürün sayfasında gösterilecek 20 yorumu belirleyiniz ve sonuçları yorumlayınız.

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# NLee the Engineer adlı yorum yapan kişi ile SkincareCEO adlı kişiyi karşılaştırdığımızda;
# SkincareCEO kişisinin total_vote değeri 1694, NLee the Engineer kişisinin total_vote değeri 1505.
# score_pos_neg_diff' e göre SkincareCEO kişisi daha yukarıda olması gerekirken incelendiğinde helpful_no değeri neredeyse
# diğer kişiye göre 2 kat daha fazla. Bu yüzden score_average_rating ve wilson_lower_bound yöntemleri daha etkili gözüküyor.

# Ancak Kelly ve Twister adlı kullanıcıları incelediğimizde;
# total_vote değerleri sırasıyla, 495 ve 49.
# score_average_rating yöntemine göre Twister adlı kullanıcı daha önce çıkıyor ancak wilson_lower_bound yöntemi bu açığı da
# yakalayarak en doğru sonucu veriyor ve Kelly adlı kullanıcıyı önce çıkarıyor çünkü daha çok yorum barındırması onun
# social proof'unu arttırıyor.
