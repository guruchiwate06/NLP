import heapq, string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

document = ("Artificial intelligence is transforming the way people live and work. "
            "Machine learning, a subset of artificial intelligence, allows computers to learn from data without being explicitly programmed. "
            "Natural language processing enables machines to read, understand and generate human language. "
            "Many companies use natural language processing for chatbots, translation and sentiment analysis. "
            "The weather today is pleasant and sunny. "
            "Deep learning models such as transformers have greatly improved the accuracy of language understanding. "
            "However, artificial intelligence also raises concerns about privacy, bias and job displacement. "
            "Researchers are working to make artificial intelligence systems fair, transparent and trustworthy.")

# 1. Sentence tokenization
sentences = sent_tokenize(document)
print("Number of sentences:", len(sentences))

# 2. Word frequency table (excluding stop words and punctuation)
stop_words = set(stopwords.words('english'))
freq = {}
for word in word_tokenize(document.lower()):
    if word not in stop_words and word not in string.punctuation:
        freq[word] = freq.get(word, 0) + 1

# 3. Normalize frequencies (weighted frequency)
max_f = max(freq.values())
weights = {w: f / max_f for w, f in freq.items()}
print("\nTop weighted words:")
for w, v in sorted(weights.items(), key=lambda x: -x[1])[:8]:
    print(f"  {w:<14}{v:.2f}")

# 4. Score each sentence = sum of its word weights
scores = {}
for i, s in enumerate(sentences):
    for w in word_tokenize(s.lower()):
        if w in weights:
            scores[i] = scores.get(i, 0) + weights[w]

print("\nSentence Scores:")
for i, s in enumerate(sentences):
    print(f"  S{i+1}: {scores.get(i, 0):.2f} | {s[:55]}...")

# 5. Select top-N sentences and keep original order
N = 3
top = sorted(heapq.nlargest(N, scores, key=scores.get))
summary = " ".join(sentences[i] for i in top)
print("\nSelected sentences:", ["S%d" % (i + 1) for i in top])
print("\nSUMMARY:\n" + summary)
o, s = len(document.split()), len(summary.split())
print(f"\nOriginal words: {o} Summary words: {s} Compression: {100 * s / o:.1f}%")

# ---------------- Visualization ----------------
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE = '#2B54A3', '#E08A1E'

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={'width_ratios': [1, 1.2]})
tw = sorted(weights.items(), key=lambda x: -x[1])[:10][::-1]
ax[0].barh([w for w, _ in tw], [v for _, v in tw], color=BLUE)
ax[0].set_xlabel('Weighted frequency')
ax[0].set_title('Top 10 word weights')
sc = [scores.get(i, 0) for i in range(len(sentences))]
b = ax[1].bar(['S%d' % (i + 1) for i in range(len(sentences))], sc,
              color=[BLUE if i in top else '#C9D1DE' for i in range(len(sentences))])
ax[1].bar_label(b, fmt='%.2f', fontsize=8, padding=2)
ax[1].set_ylabel('Sentence score')
ax[1].set_title('Sentence scores (blue = selected)')
plt.tight_layout()
plt.savefig('fig7_1.png', dpi=200)
print('\nCharts saved:', sorted(f for f in os.listdir('.') if f.startswith('fig7_')))
