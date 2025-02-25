import pandas as pd
from collections import defaultdict
import jieba
import re


# 数据清洗与分词
def clean_and_segment(text, stopwords):
    text = re.sub('<[^>]*>', '', text)  # 去除HTML标签
    text = re.sub(r'[^\w\s]', '', text)  # 去除特殊符号
    seg_list = jieba.lcut(text)
    return [word for word in seg_list if word not in stopwords]


# 加载停用词、否定词、程度副词文件
# 加载停用词、否定词、程度副词文件
def load_resources(stopwords_file, negation_file, degree_file):
    stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords.update(line.strip() for line in f)

    with open(negation_file, 'r', encoding='utf-8') as f:
        negation_words = [line.strip() for line in f]

    degree_dict = defaultdict(float)
    with open(degree_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:  # 确保有两个分段
                try:
                    degree_dict[parts[0]] = float(parts[1])
                except ValueError:
                    print(f"Warning: Line '{line.strip()}' is not in 'word,weight' format")
            else:
                print(f"Warning: Line '{line.strip()}' is malformed and will be skipped")

    return stopwords, negation_words, degree_dict


# 加载情感词典
def load_sentiment_dict(sentiment_file):
    sentiment_dict = defaultdict(float)
    with open(sentiment_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sentiment_dict[parts[0]] = float(parts[1])
    return sentiment_dict


# 分类词汇为情感词、否定词和程度副词
def classify_words(word_list, sentiment_dict, negation_words, degree_dict):
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

    return sentiment_words, negation_words_positions, degree_words_positions


# 计算加权情感得分
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


# 处理评论并计算情感分数
def preprocess_and_compute_sentiment(csv_file, stopwords_file, sentiment_file, negation_file, degree_file):
    df = pd.read_csv(csv_file)

    stopwords, negation_words, degree_dict = load_resources(stopwords_file, negation_file, degree_file)
    sentiment_dict = load_sentiment_dict(sentiment_file)

    # 清洗评论文本并去除停用词
    df['processed_text'] = df['comment'].apply(lambda text: clean_and_segment(text, stopwords))

    # 计算每条评论的情感得分
    df['sentiment_score'] = df['processed_text'].apply(
        lambda word_list: calculate_weighted_score(
            *classify_words(word_list, sentiment_dict, negation_words, degree_dict), word_list)
    )

    # 保存处理后的评论及其情感得分
    df.to_csv('processed_comments_with_sentiment.csv', index=False)

    # 计算按日期的平均情感得分
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        sentiment_by_date = df.groupby('date')['sentiment_score'].mean().reset_index()
        sentiment_by_date.columns = ['date', 'avg_sentiment_score']
        sentiment_by_date.to_csv('sentiment_avg_by_date.csv', index=False)
        print("Sentiment average score by date saved to 'sentiment_avg_by_date.csv'.")
    else:
        print("Error: The dataset does not contain a 'date' column.")

    return sentiment_by_date


# 合并情感得分与市场数据
def merge_with_market_data(sentiment_file, market_file):
    sentiment_df = pd.read_csv(sentiment_file)
    market_df = pd.read_csv(market_file, usecols=['date', 'high', 'low', 'open', 'close', 'volume'])

    # 将日期列转换为日期时间格式
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'].str.strip(), format='%Y-%m-%d')
    market_df['date'] = pd.to_datetime(market_df['date'].str.strip(), format='%Y-%m-%d %H:%M:%S')

    # 基于date列进行合并
    merged_df = pd.merge(sentiment_df, market_df, on='date')
    merged_df.to_csv('D://project//stock-forecasting//new-version//merged_file.csv', index=False)

    # 输出合并结果
    print("Merged data saved to 'merged_file.csv'.")
    print(merged_df)
    return merged_df


# 主函数
def main():
    # 文件路径
    input_csv_file = 'your_file_modified.csv'  # 评论数据文件
    stopwords_file = 'dict/stopwords.txt'  # 停用词表文件
    sentiment_file = 'dict/sentiment_dict.txt'  # 情感词典
    negation_file = 'dict/negation_words.txt'  # 否定词词典
    degree_file = 'dict/degree_words.txt'  # 程度副词词典
    market_data_file = '002603_data.csv'  # 市场数据文件

    # 处理评论并计算情感得分
    sentiment_by_date = preprocess_and_compute_sentiment(input_csv_file, stopwords_file, sentiment_file, negation_file,
                                                         degree_file)

    # 合并情感得分与市场数据
    merge_with_market_data('sentiment_avg_by_date.csv', market_data_file)


if __name__ == '__main__':
    main()