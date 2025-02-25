import pandas as pd
from collections import defaultdict
import re
import jieba


# 生成stopword表并过滤否定词和程度词汇
def load_stopwords(stopwords_file, negation_file, degree_file):
    stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as fr:
        stopwords = set(word.strip() for word in fr)
    with open(negation_file, 'r', encoding='utf-8') as fr:
        negation_words = set(word.strip() for word in fr)
    with open(degree_file, 'r', encoding='utf-8') as fr:
        degree_words = set(word.split(',')[0] for word in fr)

    return stopwords - negation_words - degree_words


# 文本预处理及分词
def clean_and_segment(text, stopwords):
    if not isinstance(text, str):
        return []  # 返回一个空列表或适合的默认值

    text = re.sub('<[^>]*>', '', text)  # 去除HTML标签
    text = re.sub(r'[^\w\s]', '', text)  # 去除特殊符号
    seg_list = jieba.cut(text)
    return [word for word in seg_list if word not in stopwords]



# 加载情感资源
def load_sentiment_resources(sentiment_file, negation_file, degree_file):
    sentiment_dict = defaultdict(float)
    with open(sentiment_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sentiment_dict[parts[0]] = float(parts[1])

    with open(negation_file, 'r', encoding='utf-8') as f:
        negation_words = [line.strip() for line in f]

    degree_dict = defaultdict(float)
    with open(degree_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                print(f"Warning: skipping malformed line in degree file: {line.strip()}")
                continue
            try:
                degree_dict[parts[0]] = float(parts[1])
            except ValueError as e:
                print(f"Warning: skipping line due to value error: {line.strip()} - {e}")
                continue

    return sentiment_dict, negation_words, degree_dict


# 分类和计算情感得分
def classify_and_score_words(word_list, sentiment_dict, negation_words, degree_dict):
    sentiment_words = {}
    negation_words_positions = {}
    degree_words_positions = {}

    for i, word in enumerate(word_list):
        if word in sentiment_dict:
            sentiment_words[i] = sentiment_dict[word]
        elif word in negation_words:
            negation_words_positions[i] = -1
        elif word in degree_dict:
            degree_words_positions[i] = degree_dict[word]

    return calculate_weighted_score(sentiment_words, negation_words_positions, degree_words_positions, word_list)


def calculate_weighted_score(sentiment_words, negation_words, degree_words, word_list):
    W = 1
    score = 0
    sentiment_indices = list(sentiment_words.keys())

    for i in range(len(word_list)):
        if i in sentiment_words:
            score += W * sentiment_words[i]
            next_sentiment_index = sentiment_indices.index(i) + 1
            if next_sentiment_index < len(sentiment_indices):
                for j in range(i + 1, sentiment_indices[next_sentiment_index]):
                    if j in negation_words:
                        W *= -1
                    elif j in degree_words:
                        W *= degree_words[j]

    return score


# 主函数：处理评论数据并合并结果
# 主函数：处理评论数据并合并结果
def preprocess_and_merge(csv_file, stopwords_file, user_dict_file, sentiment_file, negation_file, degree_file,
                         market_data_file):
    # 读取评论数据
    df = pd.read_csv(csv_file)

    # 加载资源
    stopwords = load_stopwords(stopwords_file, negation_file, degree_file)
    jieba.load_userdict(user_dict_file)
    sentiment_dict, negation_words, degree_dict = load_sentiment_resources(sentiment_file, negation_file, degree_file)

    # 清洗评论文本并去除停用词
    df['processed_text'] = df['comment'].apply(lambda text: clean_and_segment(text, stopwords))

    # 计算每条评论的情感得分
    df['sentiment_score'] = df['processed_text'].apply(
        lambda word_list: classify_and_score_words(word_list, sentiment_dict, negation_words, degree_dict)
    )

    # 保存处理后的评论及其情感得分
    df.to_csv('processed_comments_with_sentiment.csv', index=False)

    # 按日期计算情感得分的平均值
    if 'date' in df.columns:  # 确保数据中有日期列
        df['date'] = pd.to_datetime(df['date'])  # 将日期列转换为日期类型
        sentiment_by_date = df.groupby('date')['sentiment_score'].mean().reset_index()
        sentiment_by_date.columns = ['date', 'avg_sentiment_score']  # 重命名列
        sentiment_by_date.to_csv('sentiment_avg_by_date.csv', index=False)
        print("Sentiment average score by date saved to 'sentiment_avg_by_date.csv'.")

        # 修改此处，增加 'code' 列
        market_df = pd.read_csv(market_data_file, usecols=['date', 'code', 'high', 'low', 'open', 'close', 'volume'])

        # 将 market_df 的 'date' 列转换为 datetime 类型，确保与 sentiment_by_date 一致
        market_df['date'] = pd.to_datetime(market_df['date'], errors='coerce')

        # 合并 sentiment_by_date 和 market_df，保持所有 market_df 的列
        merged_df = pd.merge(sentiment_by_date, market_df, on='date')  # 默认是内连接
        merged_df.to_csv('merged_file.csv', index=False)
        print("Merged data saved to 'merged_file.csv'.")


# 运行主函数
if __name__ == "__main__":
    input_csv_file = 'your_file_modified.csv'  # 评论数据文件
    stopwords_file = 'dict/stopwords.txt'  # 停用词表文件
    user_dict_file = 'dict/user_dict.txt'  # 自定义词典
    sentiment_file = 'dict/sentiment_dict.txt'  # 情感词典
    negation_file = 'dict/negation_words.txt'  # 否定词词典
    degree_file = 'dict/degree_words.txt'  # 程度副词词典
    market_data_file = 'hs300_.csv'  # 市场数据文件

    preprocess_and_merge(input_csv_file, stopwords_file, user_dict_file, sentiment_file, negation_file, degree_file,
                         market_data_file)